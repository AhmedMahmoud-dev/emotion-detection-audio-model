# model_loader_audio.py
"""
MULTIMODAL AUDIO EMOTION DETECTION  —  v5.0 Production Pipeline
================================================================
Pipeline stages:
  1. Load & preprocess audio  (noise reduction, normalization, clipping detection)
  2. Segment with overlapping windows + Voice Activity Detection
  3. Per-segment emotion inference via emotion2vec_plus_base (funasr)
  4. Bidirectional EMA temporal smoothing (forward + backward, averaged)
  5. Transcribe audio → text (Whisper)
  6. Query text emotion API
  7. Entropy-weighted multimodal fusion (audio + text)
  8. Assemble timeline with transition detection

All outputs are native Python floats/ints — safe for FastAPI JSON serialization.
"""

# ── Environment ──────────────────────────────────────────────────────────────
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# ── Standard library ────────────────────────────────────────────────────────
import logging
import tempfile
import time
from datetime import datetime

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import requests
import soundfile as sf
import torch          # noqa: F401 – needed by funasr/whisper runtime
import librosa
import noisereduce as nr
import whisper
from funasr import AutoModel as FunASRModel

# ── Logging ─────────────────────────────────────────────────────────────────
log = logging.getLogger("audio_emotion")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_h)

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# TEXT_API_URL  = "http://127.0.0.1:8020/emotion/text_model"
# TEXT_API_URL  = "http://13.63.140.123:8000/emotion/text_model"
TEXT_API_URL  = "https://graduation-project-website-eight.vercel.app/text/emotion/text_model"



WHISPER_MODEL_NAME = "tiny"
AUDIO_MODEL_NAME   = "iic/emotion2vec_plus_base"

SAMPLE_RATE      = 16_000
WINDOW_SEC       = 2.0      # analysis window duration (seconds)
HOP_SEC          = 1.0      # hop between consecutive windows (seconds)
MIN_SEGMENT_SEC  = 0.5      # discard tail segments shorter than this

# Silence / energy thresholds (applied after normalization, in dB)
SILENCE_DB       = -35      # below this → pure silence, weight = 0
QUIET_DB         = -20      # below this → very quiet, weight = 0.3
CONFIDENCE_GATE  = 0.25     # if max(probs) < this, w *= 0.15

# Smoothing
EMA_ALPHA_BASE   = 0.30     # base EMA factor (tuned per-clip via adaptive logic)

# Fusion
TEXT_BIAS         = 1.5      # text modality priority multiplier
ENTROPY_EPS       = 0.1      # inverse-entropy epsilon (smaller = sharper weighting)
TEXT_API_TIMEOUT   = 15      # seconds

# ── Label system ────────────────────────────────────────────────────────────

FUSION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_CLASSES   = len(FUSION_LABELS)

_FUSION_INDEX = {label: i for i, label in enumerate(FUSION_LABELS)}

EMOTION_CATEGORY = {
    "anger":    "negative",
    "disgust":  "negative",
    "fear":     "negative",
    "sadness":  "negative",
    "joy":      "positive",
    "surprise": "positive",
    "neutral":  "neutral",
}

# emotion2vec_plus_base → our 7-class mapping
_E2V_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]
_E2V_TO_FUSION = {
    "angry":     "anger",
    "disgusted": "disgust",
    "fearful":   "fear",
    "happy":     "joy",
    "neutral":   "neutral",
    "other":     "neutral",
    "sad":       "sadness",
    "surprised": "surprise",
    "unknown":   "neutral",
}

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (fail-fast — no silent fallback)
# ═════════════════════════════════════════════════════════════════════════════

log.info("Loading Whisper model '%s' …", WHISPER_MODEL_NAME)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
log.info("Whisper loaded.")

log.info("Loading emotion2vec model '%s' …", AUDIO_MODEL_NAME)
audio_model = FunASRModel(model=AUDIO_MODEL_NAME, hub="hf")
log.info("emotion2vec loaded.")


# ═════════════════════════════════════════════════════════════════════════════
#  AUDIO PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def _load_and_preprocess(file_path: str):
    """
    Load audio → mono 16 kHz → noise reduce → normalize → detect clipping.
    Returns (waveform, sample_rate, quality_flags).
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    quality = {"clipping_detected": False, "original_peak": float(np.max(np.abs(y)) if len(y) > 0 else 0)}

    # Clipping detection (before any processing)
    if len(y) > 0:
        peak = np.max(np.abs(y))
        # Consecutive-max-sample run ≥ 5 is a strong clipping indicator
        if peak >= 0.999:
            clip_runs = np.diff(np.where(np.abs(y) >= 0.999)[0])
            if len(clip_runs) > 0 and np.any(clip_runs == 1):
                quality["clipping_detected"] = True
                log.warning("Clipping detected in '%s'", file_path)

    # Trim leading/trailing silence (use gentler threshold for whispered speech)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=25)

    # Noise reduction on raw dynamic range FIRST (before normalization)
    if len(y) > 0:
        y = nr.reduce_noise(y=y, sr=sr, stationary=False).astype(np.float32)

    # Single normalization pass (after NR, not before)
    if len(y) > 0:
        y = librosa.util.normalize(y)

    return y, sr, quality


# ═════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION  +  VOICE ACTIVITY DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def _build_segments(y, sr):
    """
    Sliding-window segmentation with per-segment energy analysis.
    Returns (segments, offsets, energy_weights, is_speech_flags).
    """
    win_len = int(WINDOW_SEC * sr)
    hop_len = int(HOP_SEC * sr)
    min_len = int(MIN_SEGMENT_SEC * sr)

    segments = []
    offsets  = []

    pos = 0
    while pos < len(y):
        end = min(pos + win_len, len(y))
        if (end - pos) < min_len:
            break
        segments.append(y[pos:end].astype(np.float32))
        offsets.append(pos / sr)
        if end == len(y):
            break
        pos += hop_len

    # Per-segment energy analysis
    weights   = []
    is_speech = []
    for seg in segments:
        rms = float(np.mean(librosa.feature.rms(y=seg)))
        rms_db = 20.0 * np.log10(rms + 1e-9)

        if rms_db < SILENCE_DB:
            weights.append(0.0)
            is_speech.append(False)
        elif rms_db < QUIET_DB:
            weights.append(0.3)
            is_speech.append(False)  # too quiet to be reliable speech
        else:
            weights.append(1.0)
            is_speech.append(True)

    return segments, offsets, weights, is_speech


# ═════════════════════════════════════════════════════════════════════════════
#  PER-SEGMENT EMOTION INFERENCE  (emotion2vec)
# ═════════════════════════════════════════════════════════════════════════════

def _infer_segment(seg, sr, tmp_path):
    """
    Run emotion2vec on a single segment, return 7-class probability vector.
    """
    sf.write(tmp_path, seg, sr)

    res = audio_model.generate(
        tmp_path,
        granularity="utterance",
        extract_embedding=False,
    )
    raw_scores = res[0]["scores"]

    # Map 9 → 7 classes
    probs = [0.0] * NUM_CLASSES
    for ei, e2v_label in enumerate(_E2V_LABELS):
        fusion_label = _E2V_TO_FUSION.get(e2v_label)
        if fusion_label is not None:
            probs[_FUSION_INDEX[fusion_label]] += float(raw_scores[ei])

    # Normalize
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / NUM_CLASSES] * NUM_CLASSES

    return probs


def _run_inference(segments, sr, weights, is_speech):
    """
    Run emotion inference on all segments.
    Skip segments that are pure silence (w=0) to save compute.
    Returns (raw_probs_list, weights) — weights may be mutated by confidence gate.
    """
    neutral_vec = [0.0] * NUM_CLASSES
    neutral_vec[_FUSION_INDEX["neutral"]] = 1.0

    tmp_path = os.path.join(tempfile.gettempdir(), "_emotion2vec_seg.wav")
    raw_probs = []
    final_weights = list(weights)  # copy so we can mutate

    for idx, seg in enumerate(segments):
        if final_weights[idx] <= 0.0:
            # Pure silence — assign neutral, keep weight=0
            raw_probs.append(list(neutral_vec))
            continue

        probs = _infer_segment(seg, sr, tmp_path)

        # Confidence gate: if model is very uncertain, downweight
        if max(probs) < CONFIDENCE_GATE:
            final_weights[idx] *= 0.15

        raw_probs.append(probs)

    # Cleanup temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return raw_probs, final_weights


# ═════════════════════════════════════════════════════════════════════════════
#  BIDIRECTIONAL EMA SMOOTHING
# ═════════════════════════════════════════════════════════════════════════════

def _smooth_bidirectional(raw_probs, num_segments):
    """
    Forward-backward EMA smoothing.  Averages both passes so that edge
    segments (first & last) receive equal, balanced smoothing —
    eliminating the causal-only bias of a single forward EMA.

    Alpha adapts to clip length:
      - Very short (≤5 segments):  alpha raised → less smoothing, preserve real transitions
      - Long (≥30 segments):       alpha lowered → stronger smoothing, reduce noise
    """
    if not raw_probs:
        return []

    # Adaptive alpha based on number of segments
    if num_segments <= 5:
        alpha = min(EMA_ALPHA_BASE + 0.15, 0.60)   # less smoothing for short clips
    elif num_segments >= 30:
        alpha = max(EMA_ALPHA_BASE - 0.08, 0.15)   # more smoothing for long recordings
    else:
        alpha = EMA_ALPHA_BASE

    n = len(raw_probs)

    # Forward pass
    fwd = []
    state = np.array(raw_probs[0], dtype=np.float64)
    for p in raw_probs:
        state = alpha * np.array(p) + (1.0 - alpha) * state
        s = state.sum()
        if s > 0:
            state = state / s
        fwd.append(state.copy())

    # Backward pass
    bwd = [None] * n
    state = np.array(raw_probs[-1], dtype=np.float64)
    for i in range(n - 1, -1, -1):
        state = alpha * np.array(raw_probs[i]) + (1.0 - alpha) * state
        s = state.sum()
        if s > 0:
            state = state / s
        bwd[i] = state.copy()

    # Average forward + backward
    smoothed = []
    for i in range(n):
        avg = (fwd[i] + bwd[i]) / 2.0
        s = avg.sum()
        if s > 0:
            avg = avg / s
        smoothed.append(avg.tolist())

    return smoothed


# ═════════════════════════════════════════════════════════════════════════════
#  TIMELINE ASSEMBLY  +  TRANSITION DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def _build_timeline(smoothed_probs, offsets, weights, is_speech):
    """
    Assemble per-segment timeline entries.
    Also compute weighted combined probabilities and detect emotion transitions.
    """
    timeline = []
    combined = [0.0] * NUM_CLASSES
    total_w  = 0.0
    transitions = []
    prev_dom = None

    for idx, probs in enumerate(smoothed_probs):
        w   = weights[idx]
        dom = int(np.argmax(probs))

        # Detect transitions
        dom_label = FUSION_LABELS[dom]
        if prev_dom is not None and dom_label != prev_dom:
            transitions.append({
                "from_emotion": prev_dom,
                "to_emotion":   dom_label,
                "at_offset":    round(float(offsets[idx]), 2),
                "segment_index": idx,
            })
        prev_dom = dom_label

        # Accumulate weighted combination
        for j in range(NUM_CLASSES):
            combined[j] += probs[j] * w
        total_w += w

        timeline.append({
            "segment_index":    idx,
            "timestamp_offset": round(float(offsets[idx]), 2),
            "probabilities":    {FUSION_LABELS[j]: round(float(probs[j]), 4) for j in range(NUM_CLASSES)},
            "dominant": {
                "label":      FUSION_LABELS[dom],
                "confidence": round(float(probs[dom]), 4),
                "category":   EMOTION_CATEGORY.get(FUSION_LABELS[dom]),
            },
            "intensity_weight": round(float(w), 4),
            "is_speech":        is_speech[idx],
            "frame_reference":  f"audio_seg_{idx}",
        })

    # Normalize combined probabilities
    if total_w > 0:
        combined = [c / total_w for c in combined]
    else:
        combined = [0.0] * NUM_CLASSES
        combined[_FUSION_INDEX["neutral"]] = 1.0

    return timeline, combined, transitions


# ═════════════════════════════════════════════════════════════════════════════
#  FULL AUDIO EMOTION ANALYSIS  (stages 1-4)
# ═════════════════════════════════════════════════════════════════════════════

def analyze_audio_emotion(file_path: str):
    """
    Complete audio-only emotion analysis:
      preprocess → segment → infer → smooth → timeline.
    """
    # Stage 1: load & preprocess
    y, sr, quality = _load_and_preprocess(file_path)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Stage 2: segment + VAD
    segments, offsets, weights, is_speech = _build_segments(y, sr)

    if not segments:
        log.warning("No analyzable segments in '%s' (duration=%.2fs)", file_path, duration)
        neutral_probs = [0.0] * NUM_CLASSES
        neutral_probs[_FUSION_INDEX["neutral"]] = 1.0
        return {
            "timeline":       [],
            "combined_probs": neutral_probs,
            "duration":       duration,
            "segments_count": 0,
            "transitions":    [],
            "quality":        quality,
        }

    # Stage 3: inference
    raw_probs, weights = _run_inference(segments, sr, weights, is_speech)

    # Stage 4: bidirectional smoothing
    smoothed = _smooth_bidirectional(raw_probs, len(segments))

    # Stage 5: timeline assembly
    timeline, combined, transitions = _build_timeline(smoothed, offsets, weights, is_speech)

    return {
        "timeline":       timeline,
        "combined_probs": combined,
        "duration":       duration,
        "segments_count": len(segments),
        "transitions":    transitions,
        "quality":        quality,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION  (Whisper)
# ═════════════════════════════════════════════════════════════════════════════

def transcribe_audio(file_path: str) -> str:
    result = whisper_model.transcribe(file_path)
    return result["text"]


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT EMOTION API
# ═════════════════════════════════════════════════════════════════════════════

def get_text_emotion(text: str):
    try:
        resp = requests.post(TEXT_API_URL, json={"text": text}, timeout=TEXT_API_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        log.error("Text API returned status %d", resp.status_code)
        raise ConnectionError(f"Text API returned status {resp.status_code}")
    except Exception as exc:
        log.error("Text API error: %s", exc)
        raise ConnectionError(f"Text API is offline or unreachable: {exc}") from exc


# ═════════════════════════════════════════════════════════════════════════════
#  MULTIMODAL FUSION  (Entropy-Weighted)
# ═════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(probs):
    """Shannon entropy H(p).  Higher = more uniform/uncertain."""
    arr = np.array(probs, dtype=np.float64)
    arr = arr[arr > 0]
    return float(-np.sum(arr * np.log(arr)))


def _fuse_probabilities(audio_probs, text_probs):
    """
    Entropy-aware fusion of two probability vectors.
    The more confident modality (lower entropy) receives higher weight.
    Text gets an additional bias multiplier because it distinguishes all 7 classes.
    """
    ent_a = _shannon_entropy(audio_probs)
    ent_t = _shannon_entropy(text_probs)

    w_a = 1.0 / (ent_a + ENTROPY_EPS)
    w_t = 1.0 / (ent_t + ENTROPY_EPS)
    w_t *= TEXT_BIAS

    total = w_a + w_t
    w_a /= total
    w_t /= total

    fused = [
        float(audio_probs[i] * w_a + text_probs[i] * w_t)
        for i in range(NUM_CLASSES)
    ]

    s = sum(fused)
    if s > 0:
        fused = [x / s for x in fused]

    return fused, float(w_a), float(w_t)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN MULTIMODAL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def predict_emotion_audio(file_path: str) -> dict:
    """
    Full multimodal prediction:
      audio emotion + transcription + text emotion → entropy-weighted fusion.
    """
    start = time.time()

    # 1. Transcribe
    transcribed_text = transcribe_audio(file_path)

    # 2. Text emotion
    text_result = None
    if transcribed_text and transcribed_text.strip():
        text_result = get_text_emotion(transcribed_text)

    # 3. Audio emotion (includes preprocessing + inference + smoothing + timeline)
    audio_result = analyze_audio_emotion(file_path)
    audio_probs  = audio_result["combined_probs"]

    # 4. Extract text probability vector
    text_probs7 = [0.0] * NUM_CLASSES
    text_available = False

    if text_result and "combined_results" in text_result:
        for item in text_result["combined_results"]:
            label = item.get("label")
            if label in _FUSION_INDEX:
                text_probs7[_FUSION_INDEX[label]] = float(item["confidence"])
        # Verify we actually got probabilities
        if sum(text_probs7) > 0:
            text_available = True

    if not text_available:
        # Text unavailable — use audio-only (no fusion distortion)
        text_probs7 = list(audio_probs)
        log.info("Text modality unavailable — falling back to audio-only")

    # 5. Fuse
    fused_probs, w_audio, w_text = _fuse_probabilities(audio_probs, text_probs7)
    dom = int(np.argmax(fused_probs))

    processing_ms = round((time.time() - start) * 1000, 3)

    return {
        "audio_filename": os.path.basename(file_path),
        "transcribed_text": transcribed_text,

        "audio_emotion": {
            "timeline":         audio_result["timeline"],
            "combined_probs":   [float(x) for x in audio_result["combined_probs"]],
            "segments_count":   int(audio_result["segments_count"]),
            "duration_seconds": float(audio_result["duration"]),
            "transitions":      audio_result["transitions"],
            "quality":          audio_result["quality"],
        },

        "text_emotion": text_result,

        "final_multimodal_emotion": {
            "label":              FUSION_LABELS[dom],
            "confidence":         float(fused_probs[dom]),
            "confidence_percent": round(float(fused_probs[dom]) * 100, 2),
            "category":           EMOTION_CATEGORY.get(FUSION_LABELS[dom], "neutral"),
        },

        "final_multimodal_results": [
            {
                "label":              FUSION_LABELS[i],
                "confidence":         float(fused_probs[i]),
                "confidence_percent": round(float(fused_probs[i]) * 100, 2),
            }
            for i in range(NUM_CLASSES)
        ],

        "timestamp":         datetime.now().isoformat(),
        "processing_time_ms": float(processing_ms),

        "model_info": {
            "audio_model":    AUDIO_MODEL_NAME,
            "text_model_api": TEXT_API_URL,
            "whisper_model":  WHISPER_MODEL_NAME,
            "fusion_version": "v5.0_biEMA_emotion2vec_pro",
        },
    }
