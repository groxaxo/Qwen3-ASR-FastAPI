# Quick Start Guide: Qwen3-ASR FastAPI Server

This is a quick reference guide for getting started with the Qwen3-ASR FastAPI server.

## Installation

```bash
# Install with all dependencies
pip install qwen-asr[vllm,fastapi]
```

## Start the Server

### Option 1: Direct Command (Simplest)

```bash
qwen-asr-serve-fastapi
```

Server starts at `http://0.0.0.0:8000` with default model `Qwen/Qwen3-ASR-1.7B`

### Option 2: With Environment Variables

```bash
export MODEL_ID="Qwen/Qwen3-ASR-0.6B"
export QUANT_MODE="4bit"
export PORT="9000"
qwen-asr-serve-fastapi
```

### Option 3: Using Launch Script

```bash
# Low-VRAM preset (recommended for GPUs with < 8GB)
./launch_fastapi_server.sh --low-vram

# High-performance preset
./launch_fastapi_server.sh --high-performance

# Custom configuration
./launch_fastapi_server.sh -m Qwen/Qwen3-ASR-0.6B -q none -p 9000
```

### Option 4: Docker

```bash
# Build and run
docker build -f Dockerfile.fastapi -t qwen-asr-fastapi .
docker run --gpus all -p 8000:8000 qwen-asr-fastapi

# Or use docker-compose (recommended)
docker-compose -f docker-compose.fastapi.yml up -d
```

## Test the Server

### 1. Health Check

```bash
curl http://localhost:8000/healthz
# Response: {"status":"healthy"}
```

### 2. List Models

```bash
curl http://localhost:8000/v1/models
# Response: JSON with model information
```

### 3. Transcribe Audio

```bash
# Basic transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav"

# With language specified
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=English"

# Text-only response
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=text"
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | Model to load |
| `QUANT_MODE` | `4bit` | Quantization mode (4bit, awq, gptq, none) |
| `DTYPE` | `bfloat16` | Data type (float16, bfloat16) |
| `GPU_MEMORY_UTILIZATION` | `0.8` | GPU memory to use (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Maximum context length |
| `MAX_AUDIO_SECONDS` | `1200` | Maximum audio duration (seconds) |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum file size (MB) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

### Recommended Configurations

#### For Low-VRAM Systems (< 8GB)
```bash
export MODEL_ID="Qwen/Qwen3-ASR-0.6B"
export QUANT_MODE="4bit"
export GPU_MEMORY_UTILIZATION="0.9"
export MAX_MODEL_LEN="4096"
```

#### For High-Performance Systems (>= 16GB)
```bash
export MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export QUANT_MODE="none"
export DTYPE="bfloat16"
export GPU_MEMORY_UTILIZATION="0.8"
```

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG/Opus (`.ogg`, `.opus`)

All formats are automatically converted to mono 16kHz for processing.

## Use with Open-WebUI

1. Open Open-WebUI settings
2. Add new OpenAI API connection
3. Set base URL: `http://your-server:8000/v1`
4. Leave API key empty (not required)
5. Save settings

Audio transcription will now use your Qwen3-ASR server!

## Supported Languages

30+ languages including:
- Chinese, English, Cantonese
- Spanish, French, German, Portuguese
- Japanese, Korean, Russian
- Arabic, Italian, Thai, Vietnamese
- And many more...

## Troubleshooting

### Server won't start
- Check if GPU is available: `nvidia-smi`
- Check if dependencies installed: `pip list | grep -E "fastapi|uvicorn|vllm"`

### Out of memory errors
- Reduce GPU memory: `export GPU_MEMORY_UTILIZATION="0.7"`
- Use smaller model: `export MODEL_ID="Qwen/Qwen3-ASR-0.6B"`
- Enable quantization: `export QUANT_MODE="4bit"`

### Audio processing errors
- Ensure ffmpeg is installed: `ffmpeg -version`
- Check file format is supported
- Verify file size is under limit

### 4-bit quantization not working
- This is expected with vLLM 0.14.0
- Server automatically falls back to FP16/BF16
- Check logs for actual mode used

## Getting Help

For detailed documentation, see:
- `FASTAPI_README.md` - Complete API documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- Main `README.md` - General Qwen3-ASR information

## Example Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-required"
)

# Transcribe audio
with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B",
        file=f,
        language="English"  # optional
    )
    print(transcription.text)
```

## Next Steps

1. Test with your audio files
2. Integrate with Open-WebUI
3. Adjust configuration for your GPU
4. Deploy to production with Docker

For production deployment, see Docker section in `FASTAPI_README.md`.
