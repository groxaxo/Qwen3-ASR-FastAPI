# Qwen ASR vLLM Implementation Plan

Implement a FastAPI server for Qwen ASR using vLLM, optimized for low VRAM and compatible with Open-WebUI.

## Proposed Changes

### Environment
- Use the existing `qwen-asr` Conda environment.
- Ensure `vllm` nightly and `qwen-asr` are correctly installed.

### FastAPI Server (`app.py`)
- **Startup**: Initialize `Qwen3ASRModel.LLM` with the specified `MODEL_ID` and `QUANT_MODE`.
- **Endpoints**:
  - `GET /v1/models`: List available models.
  - `POST /v1/audio/transcriptions`: Handle multipart audio uploads and return OpenAI-compatible transcriptions.
  - `GET /healthz`: Health check.
- **Quantization Logic**:
  - Try `bitsandbytes` 4-bit quantization if `QUANT_MODE` is '4bit'.
  - Fallback to `DTYPE` (e.g., bfloat16) if 4-bit is not supported by vLLM for the model.
- **Audio Processing**:
  - Use `qwen_asr` internal `normalize_audio_input` (which uses `librosa` and `soundfile`) or `ffmpeg` to ensure 16kHz mono.
- **Error Handling**: Return OpenAI-compatible error responses.

### Configuration (`.env`)
- `MODEL_ID`, `QUANT_MODE`, `DTYPE`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `MAX_AUDIO_SECONDS`.

## Verification Plan

### Automated Tests
- Run `curl` commands to verify endpoints:
  ```bash
  curl http://localhost:8000/v1/models
  curl -F file=@audio.wav http://localhost:8000/v1/audio/transcriptions
  curl http://localhost:8000/healthz
  ```
- Check logs for quantization fallback warnings.

### Manual Verification
- Verify successful transcription of a test audio file.
- Check VRAM usage with `nvidia-smi` to confirm optimization.
