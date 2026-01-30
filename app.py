import os
import time
import logging
import uuid
from typing import List, Optional, Union
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import librosa
import soundfile as sf
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-asr-server")

try:
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audio_input
except ImportError:
    logger.error(
        "qwen-asr package not found. Please install with: pip install qwen-asr[vllm]"
    )
    raise

# Environment configurations
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
QUANT_MODE = os.getenv("QUANT_MODE", "4bit")
DTYPE = os.getenv("DTYPE", "bfloat16")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "1200"))
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "False").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="Qwen ASR OpenAI-Compatible Server")

# Global model instance
model = None


def load_model():
    global model
    print(
        f"[*] Loading model: {MODEL_ID} with QUANT_MODE={QUANT_MODE}, UTIL={GPU_MEMORY_UTILIZATION}, EAGER={ENFORCE_EAGER}"
    )
    logger.info(
        f"Loading model: {MODEL_ID} with QUANT_MODE={QUANT_MODE}, UTIL={GPU_MEMORY_UTILIZATION}, EAGER={ENFORCE_EAGER}"
    )

    kwargs = {
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "enforce_eager": ENFORCE_EAGER,
    }

    # Attempt 4-bit quantization if requested
    if QUANT_MODE == "4bit":
        try:
            print("[*] Attempting to load with bitsandbytes quantization...")
            logger.info("Attempting to load with bitsandbytes quantization...")
            # We'll try passing quantization="bitsandbytes" which is supported for some models in vLLM
            model = Qwen3ASRModel.LLM(
                model=MODEL_ID,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
                **kwargs,
            )
            print("[+] Successfully loaded with bitsandbytes 4-bit quantization.")
            logger.info("Successfully loaded with bitsandbytes 4-bit quantization.")
        except Exception as e:
            msg = f"Failed to load with bitsandbytes: {e}. Falling back to {DTYPE}."
            print(f"[!] {msg}")
            logger.warning(msg)
            model = Qwen3ASRModel.LLM(model=MODEL_ID, dtype=DTYPE, **kwargs)
    else:
        print(f"[*] Loading in {DTYPE} mode...")
        model = Qwen3ASRModel.LLM(model=MODEL_ID, dtype=DTYPE, **kwargs)

    print("[+] Model loaded successfully.")
    logger.info("Model loaded successfully.")


@app.on_event("startup")
async def startup_event():
    print("[*] FastAPI server starting up...")
    import threading

    # Start model loading in a background thread to allow server to start
    threading.Thread(target=load_model, daemon=True).start()
    print("[*] Server started. Model is loading in the background...")


@app.get("/status")
async def get_status():
    return {
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "quant_mode": QUANT_MODE,
    }


# OpenAI compatibility models
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "qwen"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class TranscriptionResponse(BaseModel):
    text: str


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelInfo(id=MODEL_ID)])


@app.get("/")
async def root():
    return {
        "message": "Qwen ASR API is running",
        "endpoints": ["/v1/models", "/v1/audio/transcriptions", "/health", "/status"],
    }


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_name: str = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read file content
    try:
        content = await file.read()
        # Temporary file for librosa/soundfile if needed, but we can use io.BytesIO
        # Load and normalize audio
        import io

        audio_file = io.BytesIO(content)

        try:
            # sf.read handles file-like objects
            audio, sr = sf.read(audio_file, dtype="float32", always_2d=False)
        except Exception as e:
            logger.error(f"Error reading audio file with soundfile: {e}")
            # Fallback to librosa.load which is more robust but might be slower
            try:
                audio_file.seek(0)
                audio, sr = librosa.load(audio_file, sr=None)
            except Exception as e2:
                logger.error(f"Error reading audio file with librosa: {e2}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid audio file format: {e}, {e2}"
                )

        # Check duration
        duration = len(audio) / sr
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio duration {duration:.1f}s exceeds limit of {MAX_AUDIO_SECONDS}s",
            )

        # Process via qwen-asr
        # normalize_audio_input converts to mono 16kHz float32
        wav16k = normalize_audio_input((audio, sr))

        # Perform transcription
        results = model.transcribe(
            audio=(wav16k, SAMPLE_RATE),
            context=prompt or "",
            language=language,
            return_time_stamps=False,
        )

        if not results:
            raise HTTPException(
                status_code=500, detail="Transcription failed to produce results"
            )

        transcription_text = results[0].text

        if response_format == "text":
            return transcription_text

        return TranscriptionResponse(text=transcription_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during transcription")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": 500}},
        )


if __name__ == "__main__":
    import sys

    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except Exception as e:
        print(f"[!] Failed to start on port {PORT}: {e}")
        # Try a different port if 8000 is taken
        if PORT == 8000:
            print("[*] Trying port 8001 instead...")
            uvicorn.run(app, host=HOST, port=8001)
        else:
            sys.exit(1)
