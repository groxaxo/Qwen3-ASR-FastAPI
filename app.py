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
from subtitle_utils import format_srt_time, group_time_stamps

# Load environment variables
load_dotenv()

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
)
logging.getLogger("transformers").setLevel(logging.WARNING)
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
FORCED_ALIGNER_ID = os.getenv("FORCED_ALIGNER_ID", "")

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

    forced_aligner = FORCED_ALIGNER_ID if FORCED_ALIGNER_ID else None
    if forced_aligner:
        logger.info(f"Forced aligner enabled: {forced_aligner}")

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
                forced_aligner=forced_aligner,
                **kwargs,
            )
            print("[+] Successfully loaded with bitsandbytes 4-bit quantization.")
            logger.info("Successfully loaded with bitsandbytes 4-bit quantization.")
        except Exception as e:
            msg = f"Failed to load with bitsandbytes: {e}. Falling back to {DTYPE}."
            print(f"[!] {msg}")
            logger.warning(msg)
            model = Qwen3ASRModel.LLM(model=MODEL_ID, dtype=DTYPE, forced_aligner=forced_aligner, **kwargs)
    else:
        print(f"[*] Loading in {DTYPE} mode...")
        model = Qwen3ASRModel.LLM(model=MODEL_ID, dtype=DTYPE, forced_aligner=forced_aligner, **kwargs)

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
    srt: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[dict]] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelInfo(id=MODEL_ID)])


@app.get("/")
async def root():
    return {
        "message": "Qwen ASR API is running",
        "endpoints": ["/v1/models", "/v1/audio/transcriptions", "/health", "/healthz", "/status"],
    }


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/healthz")
async def health_check_z():
    return await health_check()


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model_name: str = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    return_timestamps: bool = Form(False),
    max_gap_sec: float = Form(0.6),
    max_chars: int = Form(40),
    split_mode: str = Form("split_by_punctuation_or_pause_or_length"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read file content
    try:
        start_time = time.time()
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
        duration_16k = round(len(wav16k) / SAMPLE_RATE, 2)

        # Use forced aligner only if the model has one loaded
        do_timestamps = return_timestamps and model.forced_aligner is not None
        if return_timestamps and not do_timestamps:
            logger.warning("return_timestamps=True but no forced aligner loaded; timestamps skipped.")

        # Perform transcription
        results = model.transcribe(
            audio=(wav16k, SAMPLE_RATE),
            context=prompt or "",
            language=language,
            return_time_stamps=do_timestamps,
        )

        if not results:
            raise HTTPException(
                status_code=500, detail="Transcription failed to produce results"
            )

        res = results[0]
        total_time = time.time() - start_time
        real_time_factor = duration_16k / total_time if total_time > 0 else 0
        logger.info(
            f"Done in {total_time:.2f}s | Audio: {duration_16k:.2f}s | RTF: {real_time_factor:.2f}x"
        )

        # Build response payload
        srt_output = None
        segments = []
        if do_timestamps and res.time_stamps:
            groups = group_time_stamps(res.time_stamps, max_gap_sec, max_chars, split_mode)
            srt_lines = []
            for i, g in enumerate(groups, 1):
                srt_lines.append(
                    f"{i}\n{format_srt_time(g['start'])} --> {format_srt_time(g['end'])}\n{g['text']}\n"
                )
                g["index"] = i
            srt_output = "\n".join(srt_lines)
            segments = groups

        transcription_text = res.text

        if response_format == "text":
            return transcription_text

        return TranscriptionResponse(
            text=transcription_text,
            srt=srt_output,
            language=res.language or None,
            duration=duration_16k,
            segments=segments or None,
        )

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
