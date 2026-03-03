# Qwen3-ASR FastAPI Server

OpenAI-compatible FastAPI server for Qwen3-ASR with vLLM backend and optimized quantization support.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's audio transcription API
- **vLLM Backend**: High-performance inference with vLLM
- **Low-VRAM Optimized**: 4-bit quantization support with automatic fallback
- **Multiple Audio Formats**: Support for wav, mp3, m4a, flac, ogg/opus
- **Automatic Audio Processing**: Converts to mono 16kHz automatically
- **Subtitle / SRT Generation**: Optionally produces `.srt`-formatted output with word-level timestamps via Qwen3-ForcedAligner
- **CJK-Aware Token Joining**: Spaces are omitted between CJK ideograph characters; Latin text is spaced correctly
- **RTF Logging**: Every transcription logs the Real-Time Factor so you can monitor GPU throughput at a glance

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

# Optional: load Qwen3-ForcedAligner for word-level timestamps and SRT output
# Leave unset (or empty) to disable — saves VRAM and startup time
export FORCED_ALIGNER_ID="Qwen/Qwen3-ForcedAligner-0.6B"

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

Both `/health` and `/healthz` are available (the `/healthz` alias is what the Docker healthcheck uses).

```bash
curl http://localhost:8000/health
# or
curl http://localhost:8000/healthz
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": "Qwen/Qwen3-ASR-1.7B",
  "device": "cuda"
}
```

A **503** is returned while the model is still loading.

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
  "language": "English",
  "duration": 3.42,
  "srt": null,
  "segments": null
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

#### With SRT Subtitles (requires `FORCED_ALIGNER_ID`)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@podcast.mp3" \
  -F "return_timestamps=true" \
  -F "max_gap_sec=0.8" \
  -F "max_chars=60" \
  -F "split_mode=split_by_punctuation_or_pause_or_length"
```

**Response:**
```json
{
  "text": "Hello world. This is a test.",
  "language": "English",
  "duration": 5.10,
  "srt": "1\n00:00:00,000 --> 00:00:01,200\nHello world.\n\n2\n00:00:01,800 --> 00:00:03,100\nThis is a test.\n",
  "segments": [
    {"index": 1, "start": 0.0, "end": 1.2, "text": "Hello world."},
    {"index": 2, "start": 1.8, "end": 3.1, "text": "This is a test."}
  ]
}
```

> **Note:** `srt` and `segments` are `null` when `return_timestamps=false` or when `FORCED_ALIGNER_ID` is not set.

#### Transcription Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | **required** | Audio file (wav, mp3, m4a, flac, ogg, opus) |
| `language` | string | `null` | Force a specific language (e.g. `Chinese`, `English`) |
| `prompt` | string | `""` | Contextual hints to improve accuracy |
| `response_format` | string | `"json"` | `"json"` or `"text"` |
| `temperature` | float | `0.0` | Sampling temperature |
| `return_timestamps` | bool | `false` | Enable word-level timestamps and SRT output (requires `FORCED_ALIGNER_ID`) |
| `max_gap_sec` | float | `0.6` | Silence gap in seconds that starts a new subtitle line |
| `max_chars` | int | `40` | Max characters per subtitle line (0 = unlimited) |
| `split_mode` | string | `"split_by_punctuation_or_pause_or_length"` | Controls when a new subtitle line starts; the string is checked for the keywords `punctuation`, `pause`, and `length` |

### Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG/Opus (`.ogg`, `.opus`)

All formats are automatically converted to mono 16kHz for processing.

## Word-Level Timestamps & SRT Subtitles

When `FORCED_ALIGNER_ID` is set, the service loads the **Qwen3-ForcedAligner-0.6B** model alongside the ASR model. Passing `return_timestamps=true` in a request will then:

1. Run forced alignment on the transcribed text to assign each word/character its exact start/end time.
2. Group the word-level timestamps into subtitle lines using the configured `max_gap_sec`, `max_chars`, and `split_mode` strategies.
3. Return an `.srt`-formatted string in the `srt` field and a structured list of segments.

The `split_mode` string is matched by substring, so a value of `"split_by_punctuation_or_pause_or_length"` (the default) enables all three splitting strategies at once. You can also use shorter strings like `"pause_length"` to enable only those two.

**CJK text handling:** Characters in the CJK Unified Ideographs block (U+4E00–U+9FFF, the most common Chinese characters) are joined without spaces between them. Latin-script text keeps normal word spacing. Hiragana, Katakana, and Hangul are not in this range, so spaces are preserved for those scripts.

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
- Health checks (via `/healthz`)
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

To enable forced-alignment subtitles in Docker:

```bash
docker run --gpus all -p 8000:8000 \
  -e FORCED_ALIGNER_ID="Qwen/Qwen3-ForcedAligner-0.6B" \
  qwen-asr-fastapi
```

Or edit `docker-compose.fastapi.yml` to set your preferred defaults.

## Logging

The server provides structured logging with timestamps:

```
[2026-01-30 12:00:00,123] [qwen-asr-server] [INFO] - Loading model: Qwen/Qwen3-ASR-1.7B with QUANT_MODE=4bit, UTIL=0.8, EAGER=False
[2026-01-30 12:00:05,456] [qwen-asr-server] [WARNING] - Failed to load with bitsandbytes: ... Falling back to bfloat16.
[2026-01-30 12:00:10,789] [qwen-asr-server] [INFO] - Model loaded successfully.
[2026-01-30 12:00:15,000] [qwen-asr-server] [INFO] - Done in 3.21s | Audio: 45.60s | RTF: 14.20x
```

The **RTF (Real-Time Factor)** line is logged after every transcription request. A value of `14.20x` means 45.6 seconds of audio were transcribed in 3.21 seconds — the higher, the faster.

Third-party library log noise (e.g. from `transformers`) is suppressed to WARNING level.

## Troubleshooting

### Model fails to load with 4-bit quantization

This is expected with vLLM 0.14.0. The server will automatically fall back to BF16/FP16:

```
WARNING - Could not enable bitsandbytes quantization
INFO - Falling back to FP16/BF16 mode
```

**Solution**: This is normal behavior. The model will still work efficiently in BF16 mode.

### `return_timestamps=true` returns no SRT output

The forced aligner is not loaded. Set `FORCED_ALIGNER_ID=Qwen/Qwen3-ForcedAligner-0.6B` in your environment and restart the server. A warning is logged if timestamps are requested but the aligner is absent.

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
