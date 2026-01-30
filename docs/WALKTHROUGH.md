# Qwen ASR vLLM Implementation Walkthrough

Successfully implemented a FastAPI server for Qwen ASR using the vLLM backend, optimized for performance and compatibility.

## Changes Made

### 1. FastAPI Server (`app.py`)
- Implemented OpenAI-style endpoints:
  - `GET /v1/models`
  - `POST /v1/audio/transcriptions`
  - `GET /health`
  - `GET /status`
- Integrated `qwen-asr` unified wrapper for vLLM.
- Added robust audio processing using `soundfile` and `librosa` fallback to ensure 16kHz mono normalization.
- Implemented background model loading to allow the server to start immediately.

### 2. Quantization & VRAM Optimization
- Added logic to attempt 4-bit `bitsandbytes` quantization for low VRAM usage.
- Implemented a graceful fallback to `bfloat16` if quantization fails or VRAM is insufficient due to other processes.
- Configurable via environment variables (`QUANT_MODE`, `GPU_MEMORY_UTILIZATION`).

### 3. Configuration
- Created `.env.example` with standard defaults for quick deployment.
- Supports `MODEL_ID`, `DTYPE`, and GPU utilization limits.

## Verification Results

### Server Status
The server successfully starts and initializes the model. Note that if other processes occupy significant VRAM, it automatically falls back to a safer loading mode.

### Transcription Verification
Verified using the official Qwen ASR test audio:
```bash
curl -X POST http://localhost:8001/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_en.wav" \
  -F "model=Qwen/Qwen3-ASR-1.7B"
```
**Result**:
`{"text": "A simple word of advice: never trust anyone."}`

### 4. Speed Test Results (Real-Time Factor)
Verified transcription speed using an 8-minute technology news video:
- **Audio Duration**: 488.01s (~8.1 minutes)
- **Processing Time**: 13.71s
- **Real-Time Factor (RTF)**: **0.0281** (35.6x faster than real-time)

## Usage Instructions
1. Activate Conda environment: `conda activate qwen-asr`
2. Set environment variables in `.env`.
3. Run server: `python app.py`
4. Access API at `http://localhost:8001` (or configured port).
5. Push to GitHub: `git add . && git commit -m "Add Qwen ASR FastAPI server with vLLM" && git push`
