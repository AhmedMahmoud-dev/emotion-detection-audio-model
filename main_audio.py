# main_audio.py
"""
FastAPI entry point for the Audio Emotion Detection microservice.
"""

import logging
import os
import time

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_loader_audio import predict_emotion_audio

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audio_api")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Emotion Detection API",
    version="5.0.0",
    description="Multimodal audio emotion analysis with timeline tracking",
)

# --- CORS Configuration ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- End CORS Configuration ---

TMP_DIR = "tmp_audio"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mpeg", ".webm"}
MAX_FILE_SIZE_MB = 50


@app.post("/emotion/audio_model")
async def audio_emotion_api(file: UploadFile = File(...)):
    if file is None or file.filename is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    os.makedirs(TMP_DIR, exist_ok=True)
    safe_name = f"{int(time.time() * 1000)}_{file.filename}"
    path = os.path.join(TMP_DIR, safe_name)

    # Read and validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    with open(path, "wb") as f:
        f.write(contents)

    log.info("Processing '%s' (%.2f MB)", file.filename, size_mb)

    try:
        result = predict_emotion_audio(path)
        return result
    except Exception as exc:
        log.exception("Pipeline failed for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.get("/")
def home():
    return {"status": "Audio Emotion API is running", "version": "5.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main_audio:app", host="0.0.0.0", port=8001)
