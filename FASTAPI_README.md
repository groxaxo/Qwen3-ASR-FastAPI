# Qwen3-ASR FastAPI Server

OpenAI-compatible FastAPI server for Qwen3-ASR with vLLM backend and optimized quantization support.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's audio transcription API
- **vLLM Backend**: High-performance inference with vLLM
- **Low-VRAM Optimized**: 4-bit quantization support with automatic fallback
- **Multiple Audio Formats**: Support for wav, mp3, m4a, flac, ogg/opus
- **Automatic Audio Processing**: Converts to mono 16kHz automatically

## Installation

```bash
# Install with FastAPI and vLLM support
pip install qwen-asr[vllm,fastapi]

# Or install all optional dependencies
pip install -e ".[vllm,fastapi]"
```

## Quick Start

### Basic Usage

```bash
# Start the server with default settings
qwen-asr-serve-fastapi
```

The server will start on `http://0.0.0.0:8000` with the default model `Qwen/Qwen3-ASR-1.7B`.

### Environment Variables

Configure the server using environment variables:

```bash
# Model configuration
export MODEL_ID="Qwen/Qwen3-ASR-1.7B"  # or Qwen/Qwen3-ASR-0.6B
export QUANT_MODE="4bit"                # 4bit, awq, gptq, or none
export DTYPE="bfloat16"                 # float16 or bfloat16
export GPU_MEMORY_UTILIZATION="0.8"     # 0.0 to 1.0
export MAX_MODEL_LEN="8192"             # Maximum context length

# Audio processing limits
export MAX_AUDIO_SECONDS="1200"         # Maximum audio duration (20 minutes)
export MAX_UPLOAD_SIZE_MB="100"         # Maximum file upload size

# Server configuration
export HOST="0.0.0.0"
export PORT="8000"

# Start the server
qwen-asr-serve-fastapi
```

## Quantization Modes

### 4-bit Quantization (Recommended for Low-VRAM)

```bash
export QUANT_MODE="4bit"
export DTYPE="bfloat16"
qwen-asr-serve-fastapi
```

**Note**: 4-bit quantization support in vLLM 0.14.0 is limited. The server will:
1. Attempt to load with bitsandbytes quantization if supported
2. Automatically fall back to FP16/BF16 if 4-bit is not available
3. Log clear warnings about the quantization mode being used

### AWQ/GPTQ Quantization

If your model checkpoint has AWQ or GPTQ quantization:

```bash
export QUANT_MODE="awq"  # or "gptq"
qwen-asr-serve-fastapi
```

### No Quantization (Full Precision)

```bash
export QUANT_MODE="none"
export DTYPE="bfloat16"
qwen-asr-serve-fastapi
```

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/healthz
```

**Response:**
```json
{
  "status": "healthy"
}
```

### 2. List Models

```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-ASR-1.7B",
      "object": "model",
      "created": 1706659200,
      "owned_by": "qwen"
    }
  ]
}
```

### 3. Audio Transcription

#### Basic Transcription

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "English"
}
```

#### With Language Specification

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=Chinese"
```

#### With Context Prompt

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "prompt=Previous context or style guide"
```

#### Text-only Response

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

**Response:**
```
Hello, this is a test transcription.
```

### Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG/Opus (`.ogg`, `.opus`)

All formats are automatically converted to mono 16kHz for processing.

## Open-WebUI Integration

This server is fully compatible with Open-WebUI. Configure Open-WebUI to use this server as an OpenAI-compatible endpoint:

1. In Open-WebUI settings, add a new OpenAI API connection
2. Set the base URL to `http://your-server:8000/v1`
3. API key is not required (leave empty or use any value)
4. The audio transcription feature will automatically use this endpoint

## Error Handling

The API returns OpenAI-compatible error responses:

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": null
  }
}
```

### Common Errors

- **413**: File too large (exceeds `MAX_UPLOAD_SIZE_MB`)
- **400**: Invalid file format or audio too long (exceeds `MAX_AUDIO_SECONDS`)
- **400**: Audio too short (< 0.1 seconds)
- **503**: Model not loaded or service unavailable

## Performance Optimization

### For Low-VRAM Systems (< 8GB)

```bash
export MODEL_ID="Qwen/Qwen3-ASR-0.6B"   # Use smaller model
export QUANT_MODE="4bit"                 # Enable quantization
export GPU_MEMORY_UTILIZATION="0.9"      # Use more GPU memory
export MAX_MODEL_LEN="4096"              # Reduce context length
```

### For High-Performance Systems

```bash
export MODEL_ID="Qwen/Qwen3-ASR-1.7B"   # Use larger model
export QUANT_MODE="none"                 # No quantization
export DTYPE="bfloat16"                  # BF16 for better performance
export GPU_MEMORY_UTILIZATION="0.8"
```

## Docker Deployment

### Using Dockerfile

A pre-configured Dockerfile is provided at `Dockerfile.fastapi`:

```bash
# Build the image
docker build -f Dockerfile.fastapi -t qwen-asr-fastapi .

# Run the container
docker run --gpus all -p 8000:8000 qwen-asr-fastapi
```

### Using Docker Compose (Recommended)

For easier deployment with docker-compose:

```bash
# Start the service
docker-compose -f docker-compose.fastapi.yml up -d

# View logs
docker-compose -f docker-compose.fastapi.yml logs -f

# Stop the service
docker-compose -f docker-compose.fastapi.yml down
```

The docker-compose configuration includes:
- GPU support
- Automatic restart
- Health checks
- HuggingFace model caching
- Environment variable configuration

### Custom Configuration with Docker

Override environment variables when running:

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_ID="Qwen/Qwen3-ASR-0.6B" \
  -e QUANT_MODE="none" \
  -e GPU_MEMORY_UTILIZATION="0.9" \
  qwen-asr-fastapi
```

Or edit `docker-compose.fastapi.yml` to set your preferred defaults.

## Logging

The server provides detailed logging including:
- Model loading progress
- Quantization mode selection and fallback information
- Request processing details
- Error messages

Logs are written to stdout in this format:
```
2026-01-30 12:00:00 - qwen_asr.cli.serve_fastapi - INFO - Loading model: Qwen/Qwen3-ASR-1.7B
2026-01-30 12:00:00 - qwen_asr.cli.serve_fastapi - INFO - Quantization mode: 4bit
2026-01-30 12:00:05 - qwen_asr.cli.serve_fastapi - WARNING - ⚠ Model loaded in fallback mode with dtype=bfloat16 (quantization not available)
```

## Troubleshooting

### Model fails to load with 4-bit quantization

This is expected with vLLM 0.14.0. The server will automatically fall back to BF16/FP16:

```
WARNING - Could not enable bitsandbytes quantization
INFO - Falling back to FP16/BF16 mode
```

**Solution**: This is normal behavior. The model will still work efficiently in BF16 mode.

### Out of memory errors

Reduce GPU memory usage:

```bash
export GPU_MEMORY_UTILIZATION="0.7"
export MAX_MODEL_LEN="4096"
# Or use the smaller model
export MODEL_ID="Qwen/Qwen3-ASR-0.6B"
```

### Audio processing errors

Ensure ffmpeg is installed:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Supported Languages

The model supports 30 languages and 22 Chinese dialects including:
- Chinese, English, Cantonese
- Arabic, German, French, Spanish, Portuguese
- Indonesian, Italian, Korean, Russian
- Thai, Vietnamese, Japanese, Turkish
- And many more...

See the main README for the complete list.

## License

Apache-2.0
