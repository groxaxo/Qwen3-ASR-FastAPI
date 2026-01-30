# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FastAPI server for Qwen3-ASR with OpenAI-compatible endpoints.

This server provides:
- vLLM-based inference with quantization support
- OpenAI-compatible audio transcription API
- Health check endpoint
- Model listing endpoint

Environment Variables:
    MODEL_ID: HuggingFace model ID (default: Qwen/Qwen3-ASR-1.7B)
    QUANT_MODE: Quantization mode (4bit, awq, gptq, none) (default: 4bit)
    DTYPE: Data type (float16, bfloat16) (default: bfloat16)
    GPU_MEMORY_UTILIZATION: GPU memory utilization (default: 0.8)
    MAX_MODEL_LEN: Maximum model length (default: 8192)
    MAX_AUDIO_SECONDS: Maximum audio duration in seconds (default: 1200)
    MAX_UPLOAD_SIZE_MB: Maximum upload size in MB (default: 100)
    HOST: Server host (default: 0.0.0.0)
    PORT: Server port (default: 8000)
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
QUANT_MODE = os.getenv("QUANT_MODE", "4bit")
DTYPE = os.getenv("DTYPE", "bfloat16")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "1200"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Global model instance
asr_model = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response"""
    error: dict


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "qwen"


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint"""
    object: str = "list"
    data: list[ModelInfo]


class TranscriptionResponse(BaseModel):
    """Response for /v1/audio/transcriptions endpoint"""
    text: str
    language: Optional[str] = None


def create_error_response(message: str, error_type: str = "invalid_request_error", status_code: int = 400):
    """Create OpenAI-compatible error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": None
            }
        }
    )


async def load_model():
    """Load the ASR model with vLLM backend and quantization support"""
    global asr_model
    
    logger.info(f"Loading model: {MODEL_ID}")
    logger.info(f"Quantization mode: {QUANT_MODE}")
    logger.info(f"Data type: {DTYPE}")
    logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
        
        # Prepare vLLM arguments
        vllm_kwargs = {
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "dtype": DTYPE,
        }
        
        # Configure quantization
        if QUANT_MODE.lower() not in ["none", "no", "false", ""]:
            if QUANT_MODE.lower() in ["4bit", "bnb", "bitsandbytes"]:
                # Try to use bitsandbytes quantization
                try:
                    # Check if bitsandbytes quantization is supported in vLLM
                    # vLLM 0.14.0 has limited quantization support
                    # We'll attempt to load with quantization and fallback if needed
                    logger.warning(
                        f"4-bit quantization requested, but vLLM {vllm_kwargs.get('version', '0.14.0')} "
                        "has limited native 4-bit support. Attempting to load model with available quantization."
                    )
                    # Try loading with bitsandbytes if model supports it
                    vllm_kwargs["quantization"] = "bitsandbytes"
                except Exception as e:
                    logger.warning(f"Could not enable bitsandbytes quantization: {e}")
                    logger.info("Falling back to FP16/BF16 mode")
                    # Remove quantization parameter, will use dtype instead
                    vllm_kwargs.pop("quantization", None)
            elif QUANT_MODE.lower() in ["awq", "gptq"]:
                vllm_kwargs["quantization"] = QUANT_MODE.lower()
                logger.info(f"Using {QUANT_MODE.upper()} quantization")
            else:
                logger.warning(f"Unknown quantization mode: {QUANT_MODE}, falling back to dtype={DTYPE}")
        else:
            logger.info(f"Quantization disabled, using dtype={DTYPE}")
        
        # Load the model
        try:
            asr_model = Qwen3ASRModel.LLM(
                model=MODEL_ID,
                **vllm_kwargs
            )
            logger.info("Model loaded successfully with vLLM backend")
            
            # Log actual configuration
            if "quantization" in vllm_kwargs:
                logger.info(f"✓ Quantization mode: {vllm_kwargs['quantization']}")
            else:
                logger.info(f"✓ Quantization: None (using dtype={DTYPE})")
                
        except Exception as e:
            # Fallback: try without quantization
            if "quantization" in vllm_kwargs:
                logger.error(f"Failed to load model with quantization: {e}")
                logger.info("Attempting fallback: loading without quantization")
                vllm_kwargs_fallback = vllm_kwargs.copy()
                vllm_kwargs_fallback.pop("quantization", None)
                
                asr_model = Qwen3ASRModel.LLM(
                    model=MODEL_ID,
                    **vllm_kwargs_fallback
                )
                logger.warning(f"⚠ Model loaded in fallback mode with dtype={DTYPE} (quantization not available)")
            else:
                raise
                
    except ImportError as e:
        logger.error("Failed to import required dependencies. Make sure to install: pip install qwen-asr[vllm]")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down server")


# Create FastAPI app
app = FastAPI(
    title="Qwen3-ASR FastAPI Server",
    description="OpenAI-compatible ASR API powered by vLLM",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    if asr_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models (OpenAI-compatible)"""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=MODEL_ID,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
):
    """
    Transcribe audio file (OpenAI-compatible)
    
    Args:
        file: Audio file to transcribe
        model: Model to use (optional, uses server default)
        language: Language of the audio (optional, auto-detected if not provided)
        prompt: Optional text to guide the model's style or continue a previous segment
        response_format: Response format (json or text)
        temperature: Sampling temperature (0-1, optional)
    """
    if asr_model is None:
        return create_error_response(
            "Model not loaded",
            error_type="service_unavailable",
            status_code=503
        )
    
    # Validate response format
    if response_format not in ["json", "text"]:
        return create_error_response(
            f"Invalid response_format: {response_format}. Must be 'json' or 'text'.",
            error_type="invalid_request_error",
            status_code=400
        )
    
    # Read and validate file
    try:
        audio_bytes = await file.read()
        
        # Check file size
        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            return create_error_response(
                f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({MAX_UPLOAD_SIZE_MB}MB)",
                error_type="invalid_request_error",
                status_code=413
            )
        
        # Check file format by extension
        filename = file.filename or ""
        supported_formats = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"]
        if not any(filename.lower().endswith(ext) for ext in supported_formats):
            return create_error_response(
                f"Unsupported file format. Supported formats: {', '.join(supported_formats)}",
                error_type="invalid_request_error",
                status_code=400
            )
        
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return create_error_response(
            "Failed to read audio file",
            error_type="invalid_request_error",
            status_code=400
        )
    
    # Convert audio to required format and check duration
    try:
        # For non-WAV files, we need to convert them
        # Save to temporary file for ffmpeg/soundfile processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Load audio using soundfile/librosa (via normalize_audio_input)
            from qwen_asr.inference.utils import normalize_audio_input, SAMPLE_RATE
            
            # Load and normalize audio to mono 16kHz
            audio_np = normalize_audio_input(tmp_path)
            
            # Check duration
            duration_sec = len(audio_np) / SAMPLE_RATE
            if duration_sec > MAX_AUDIO_SECONDS:
                return create_error_response(
                    f"Audio duration ({duration_sec:.1f}s) exceeds maximum allowed duration ({MAX_AUDIO_SECONDS}s)",
                    error_type="invalid_request_error",
                    status_code=400
                )
            
            if duration_sec < 0.1:
                return create_error_response(
                    "Audio is too short (minimum 0.1 seconds)",
                    error_type="invalid_request_error",
                    status_code=400
                )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return create_error_response(
            f"Failed to process audio file: {str(e)}",
            error_type="invalid_request_error",
            status_code=400
        )
    
    # Transcribe
    try:
        # Prepare transcription parameters
        transcription_kwargs = {
            "audio": (audio_np, SAMPLE_RATE),
            "return_time_stamps": False,
        }
        
        # Add optional parameters
        if language:
            transcription_kwargs["language"] = language
        if prompt:
            transcription_kwargs["context"] = prompt
        
        # Run transcription
        results = asr_model.transcribe(**transcription_kwargs)
        
        if not results or len(results) == 0:
            return create_error_response(
                "Transcription failed: no results returned",
                error_type="server_error",
                status_code=500
            )
        
        result = results[0]
        
        # Format response
        if response_format == "text":
            return result.text
        else:
            # JSON format
            response_data = {
                "text": result.text
            }
            if result.language:
                response_data["language"] = result.language
            
            return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return create_error_response(
            f"Transcription failed: {str(e)}",
            error_type="server_error",
            status_code=500
        )


def main():
    """Main entry point for the FastAPI server"""
    import uvicorn
    
    logger.info(f"Starting Qwen3-ASR FastAPI server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Max audio duration: {MAX_AUDIO_SECONDS}s")
    logger.info(f"Max upload size: {MAX_UPLOAD_SIZE_MB}MB")
    
    uvicorn.run(
        "qwen_asr.cli.serve_fastapi:app",
        host=HOST,
        port=PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
