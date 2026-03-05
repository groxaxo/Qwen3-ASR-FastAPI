<div align="center">

<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/logo.png" width="360" alt="Qwen3-ASR"/>

# Qwen3-ASR FastAPI Server

**OpenAI-compatible ASR server · vLLM-accelerated · up to 80× real-time · Open-WebUI ready**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/qwen-asr?color=orange&logo=pypi&logoColor=white)](https://pypi.org/project/qwen-asr/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![vLLM](https://img.shields.io/badge/vLLM-0.14.0-blueviolet.svg)](https://github.com/vllm-project/vllm)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile.fastapi)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference/audio)

<p>
&nbsp;&nbsp;🤗&nbsp;<a href="https://huggingface.co/collections/Qwen/qwen3-asr">Hugging Face</a>&nbsp;&nbsp;|&nbsp;&nbsp;
🤖&nbsp;<a href="https://modelscope.cn/collections/Qwen/Qwen3-ASR">ModelScope</a>&nbsp;&nbsp;|&nbsp;&nbsp;
📑&nbsp;<a href="https://qwen.ai/blog?id=qwen3asr">Blog</a>&nbsp;&nbsp;|&nbsp;&nbsp;
📄&nbsp;<a href="https://arxiv.org/abs/2601.21337">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;
🖥️&nbsp;<a href="https://huggingface.co/spaces/Qwen/Qwen3-ASR">HF Demo</a>&nbsp;&nbsp;|&nbsp;&nbsp;
🫨&nbsp;<a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

</div>

---

This repository provides an **optimized FastAPI server** for [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) with a drop-in OpenAI `/v1/audio/transcriptions` endpoint backed by vLLM. It is a fork of [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) with production-ready serving additions.

> **2026-01-29** 🎉 Qwen3-ASR series (0.6B / 1.7B) and Qwen3-ForcedAligner-0.6B released. See the [blog post](https://qwen.ai/blog?id=qwen3asr).

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ **OpenAI-Compatible API** | Drop-in `/v1/audio/transcriptions` — works with any OpenAI client |
| ⚡ **vLLM Backend** | Up to **80× real-time** transcription throughput |
| 🗜️ **Quantization** | 4-bit / AWQ / GPTQ with automatic BF16 fallback |
| 🌍 **52 Languages** | 30 languages + 22 Chinese dialects |
| 🎵 **All Audio Formats** | WAV, MP3, M4A, FLAC, OGG/Opus — auto-converted to 16 kHz mono |
| 🖥️ **Open-WebUI Ready** | Works as an OpenAI speech endpoint without an API key |
| 🐳 **Docker + Compose** | Production-ready containers with GPU support & health checks |
| ⏱️ **SRT Subtitles** | Word-level forced alignment → SRT subtitles via `return_timestamps=true` |
| 📺 **Streaming** | Real-time streaming inference via vLLM |


## 📦 Models

| Model | Params | Languages | Inference | VRAM (FP16) |
|---|---|---|---|---|
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | 30 lang + 22 dialects | Offline / Streaming | ~22 GB |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen/Qwen3-ASR-0.6B) | 0.6B | 30 lang + 22 dialects | Offline / Streaming | ~7 GB (util=0.3) |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | 0.6B | 11 languages | NAR (timestamps) | — |


## Contents

- [Quick Start (5 min)](#-quick-start-5-min)
- [Installation](#-installation)
  - [pip](#pip)
  - [conda](#conda)
  - [From source](#from-source)
  - [Docker](#docker-1)
  - [Flash Attention 2 (optional)](#flash-attention-2-optional)
- [FastAPI Server](#-fastapi-server)
  - [Start the server](#start-the-server)
  - [Configuration reference](#configuration-reference)
  - [API endpoints](#api-endpoints)
  - [Quantization modes](#quantization-modes)
  - [Performance presets](#performance-presets)
  - [Open-WebUI integration](#open-webui-integration)
- [Docker Deployment](#-docker-deployment)
- [Python Package Usage](#-python-package-usage)
  - [Quick inference (Transformers)](#quick-inference-transformers-backend)
  - [vLLM backend](#vllm-backend)
  - [Streaming inference](#streaming-inference)
  - [ForcedAligner](#forcedaligner-usage)
- [Native vLLM Serving](#-native-vllm-serving)
- [DashScope API](#-dashscope-api)
- [Local Web UI Demo](#-local-web-ui-demo)
- [Fine-tuning](#-fine-tuning)
- [Model Architecture](#-model-architecture)
- [Supported Languages & Dialects](#-supported-languages--dialects)
- [Evaluation](#-evaluation)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)


## 🚀 Quick Start (5 min)

```bash
# 1. Install
pip install qwen-asr[vllm,fastapi]

# 2. Start the server (downloads model on first run)
qwen-asr-serve-fastapi

# 3. Health check
curl http://localhost:8000/healthz

# 4. Transcribe audio
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav"

# 5. Use from Python (OpenAI client)
python - <<'EOF'
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-required")
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="Qwen/Qwen3-ASR-1.7B", file=f)
print(result.text)
EOF
```


## 📦 Installation

### pip

Minimal install (Transformers backend only):

```bash
pip install -U qwen-asr
```

With vLLM backend (recommended for serving):

```bash
pip install -U qwen-asr[vllm]
```

Full install (FastAPI server + vLLM):

```bash
pip install -U qwen-asr[vllm,fastapi]
```

### conda

A ready-to-use conda environment file is included:

```bash
# Create the environment (installs Python 3.12, ffmpeg, and all deps)
conda env create -f environment.yml

# Activate
conda activate qwen3-asr

# Start the server
qwen-asr-serve-fastapi
```

Or create the environment manually:

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
conda install -c conda-forge ffmpeg -y
pip install -U qwen-asr[vllm,fastapi]
```

### From source

```bash
git clone https://github.com/groxaxo/Qwen3-ASR-FastAPI.git
cd Qwen3-ASR-FastAPI
pip install -e ".[vllm,fastapi]"
```

### Docker

See [Docker Deployment](#-docker-deployment) for the full guide.

### Flash Attention 2 (optional)

Reduces GPU memory usage and speeds up inference for long audio and large batch sizes. Requires compatible hardware (Ampere+).

```bash
pip install -U flash-attn --no-build-isolation
# On machines with limited RAM (< 96 GB) and many CPU cores:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Flash Attention 2 requires the model to be loaded in `float16` or `bfloat16`.


## 🖥️ FastAPI Server

The FastAPI server is the primary deployment mode for this repository. It exposes an **OpenAI-compatible** `/v1/audio/transcriptions` endpoint backed by vLLM, making it a drop-in replacement for the OpenAI Whisper API.

### Start the server

**Option 1 — Direct command (simplest)**

```bash
qwen-asr-serve-fastapi
```

**Option 2 — Environment variables**

```bash
export MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export QUANT_MODE="4bit"
export PORT="8000"
qwen-asr-serve-fastapi
```

**Option 3 — Launch script with presets**

```bash
# Low-VRAM preset (< 8 GB GPU)
./launch_fastapi_server.sh --low-vram

# High-performance preset
./launch_fastapi_server.sh --high-performance

# Custom
./launch_fastapi_server.sh -m Qwen/Qwen3-ASR-0.6B -q none -p 9000
```

**Option 4 — Docker Compose**

```bash
docker compose -f docker-compose.fastapi.yml up -d
```

### Configuration reference

All settings are controlled via environment variables (also supported in `.env` — copy `.env.example` to `.env`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID |
| `QUANT_MODE` | `4bit` | `4bit` · `awq` · `gptq` · `none` |
| `DTYPE` | `bfloat16` | `float16` · `bfloat16` |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM KV-cache pool fraction (0–1) |
| `MAX_MODEL_LEN` | `8192` | Maximum context length in tokens |
| `MAX_AUDIO_SECONDS` | `1200` | Maximum audio duration (seconds) |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum file upload size (MB) |
| `ENFORCE_EAGER` | `false` | Disable CUDA graphs (use for debug / very low VRAM) |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Reduce GPU memory fragmentation |
| `FORCED_ALIGNER_ID` | *(empty)* | HuggingFace ID for ForcedAligner (enables SRT/timestamps) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `CUDA_VISIBLE_DEVICES` | *(system default)* | GPU device selection |

### API endpoints

#### Health check

```bash
curl http://localhost:8000/healthz
# {"status": "healthy"}
```

#### List models

```bash
curl http://localhost:8000/v1/models
```

#### Transcribe audio

```bash
# Basic transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav"

# Specify language (skips auto-detection)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=English"

# With a context prompt
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "prompt=Medical terminology expected"

# Plain-text response
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"

# With SRT subtitles (requires FORCED_ALIGNER_ID to be set)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "return_timestamps=true"
```

**Request form fields:**

| Field | Default | Description |
|---|---|---|
| `file` | required | Audio file (WAV/MP3/M4A/FLAC/OGG/Opus) |
| `language` | `null` | Force language (e.g. `English`, `Chinese`); `null` = auto-detect |
| `prompt` | `null` | Context hint / style guide |
| `response_format` | `json` | `json` or `text` |
| `return_timestamps` | `false` | Enable SRT + segments (requires `FORCED_ALIGNER_ID`) |
| `max_gap_sec` | `0.6` | Max silence gap before new subtitle segment |
| `max_chars` | `40` | Max characters per subtitle line |
| `split_mode` | `split_by_punctuation_or_pause_or_length` | Subtitle split strategy |

**Supported audio formats:** WAV · MP3 · M4A · FLAC · OGG · Opus  
All formats are automatically converted to 16 kHz mono before processing.

**JSON response:**
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "English",
  "duration": 3.2,
  "srt": "1\n00:00:00,000 --> 00:00:03,200\nHello, this is a test transcription.\n",
  "segments": [{"index": 1, "start": 0.0, "end": 3.2, "text": "Hello, this is a test transcription."}]
}
```

> `srt` and `segments` are `null` when `return_timestamps=false` or no ForcedAligner is configured.

**Error responses** follow the OpenAI format:
```json
{"error": {"message": "...", "type": "invalid_request_error", "code": null}}
```

HTTP status codes: `400` (bad request / unsupported format / too long), `413` (file too large), `503` (model not ready).

#### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-required")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B",
        file=f,
        language="English",  # optional
    )
print(result.text)
```

### Quantization modes

| Mode | VRAM | Notes |
|---|---|---|
| `4bit` | Lowest | Attempts bitsandbytes; **auto-falls back to BF16** if unsupported by vLLM |
| `awq` | Low | For AWQ-quantized model checkpoints |
| `gptq` | Low | For GPTQ-quantized model checkpoints |
| `none` | Full | FP16 / BF16 — best accuracy and speed |

> **Note on vLLM 0.14.0:** Native 4-bit quantization support is limited. The server automatically falls back to BF16/FP16 with a clear log message (`⚠ Model loaded in fallback mode`). This is expected behaviour and the model remains fully functional.

### Performance presets

**Low-VRAM (< 8 GB GPU)**

```bash
export MODEL_ID="Qwen/Qwen3-ASR-0.6B"
export QUANT_MODE="4bit"
export GPU_MEMORY_UTILIZATION="0.9"
export MAX_MODEL_LEN="4096"
qwen-asr-serve-fastapi
```

**High-performance (≥ 16 GB GPU)**

```bash
export MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export QUANT_MODE="none"
export DTYPE="bfloat16"
export GPU_MEMORY_UTILIZATION="0.8"
qwen-asr-serve-fastapi
```

### Open-WebUI integration

1. Open **Settings → Connections** in Open-WebUI
2. Add a new OpenAI API connection
3. Set **Base URL** to `http://your-server:8000/v1`
4. Leave **API key** empty (not required)
5. Save — audio transcription will now use your Qwen3-ASR server


## 🐳 Docker Deployment

### Using Dockerfile

```bash
# Build
docker build -f Dockerfile.fastapi -t qwen3-asr-fastapi .

# Run
docker run --gpus all -p 8000:8000 qwen3-asr-fastapi

# Custom config
docker run --gpus all -p 8000:8000 \
  -e MODEL_ID="Qwen/Qwen3-ASR-0.6B" \
  -e QUANT_MODE="none" \
  -e GPU_MEMORY_UTILIZATION="0.9" \
  qwen3-asr-fastapi
```

### Using Docker Compose (recommended)

```bash
# Start
docker compose -f docker-compose.fastapi.yml up -d

# View logs
docker compose -f docker-compose.fastapi.yml logs -f

# Stop
docker compose -f docker-compose.fastapi.yml down
```

The Compose file includes:
- GPU reservation (NVIDIA)
- `unless-stopped` restart policy
- HuggingFace model cache volume mount
- Health check (`/healthz`)

### Official Qwen3-ASR base image

For interactive development / research use, the official `qwenllm/qwen3-asr:latest` image is available on Docker Hub:

```bash
LOCAL_WORKDIR=/path/to/your/workspace
docker run --gpus all --name qwen3-asr \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8000:80 \
    --mount type=bind,source=$LOCAL_WORKDIR,target=/data/shared/Qwen3-ASR \
    --shm-size=4gb \
    -it qwenllm/qwen3-asr:latest
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## 🐍 Python Package Usage

### Quick inference (Transformers backend)

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None,  # None = auto-detect
)
print(results[0].language)
print(results[0].text)
```

With timestamps (requires ForcedAligner):

```python
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
)

results = model.transcribe(
    audio=["path/to/audio_zh.wav", "path/to/audio_en.wav"],
    language=["Chinese", "English"],
    return_time_stamps=True,
)
for r in results:
    print(r.language, r.text, r.time_stamps[0])
```

### vLLM backend

```python
from qwen_asr import Qwen3ASRModel

if __name__ == "__main__":
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128,
        max_new_tokens=4096,
    )

    results = model.transcribe(
        audio=[
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
        language=["Chinese", "English"],
    )
    for r in results:
        print(r.language, r.text)
```

> ⚠️ Wrap code in `if __name__ == "__main__":` to avoid vLLM multiprocessing errors.

### Streaming inference

Streaming is available with the vLLM backend only (no batch or timestamps support in streaming mode). See [`examples/example_qwen3_asr_vllm_streaming.py`](examples/example_qwen3_asr_vllm_streaming.py) for details.

### ForcedAligner usage

```python
import torch
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.align(
    audio="path/to/audio_zh.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)
for token in results[0]:
    print(token.text, token.start_time, token.end_time)
```


## 🌐 Native vLLM Serving

vLLM natively supports Qwen3-ASR with `vllm serve`. Use this for maximum control or multi-GPU setups.

### Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"
```

### Start the server

```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```

### Query the server

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}]}]}'
```

Or use the `qwen-asr-serve` wrapper:

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --port 8000
```

### Offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-ASR-1.7B")
sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

outputs = llm.chat([{
    "role": "user",
    "content": [{
        "type": "audio_url",
        "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}
    }]
}], sampling_params)
print(outputs[0].outputs[0].text)
```


## 📡 DashScope API

For a hosted API option, use the Alibaba DashScope API:

| API | China | International |
|---|---|---|
| Real-time ASR | [link](https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition) | [link](https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition) |
| FileTrans ASR | [link](https://help.aliyun.com/zh/model-studio/qwen-speech-recognition) | [link](https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition) |


## 🖥️ Local Web UI Demo

### Gradio demo

```bash
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization": 0.7}' \
  --ip 0.0.0.0 --port 8000
```

With timestamps:

```bash
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000
```

For HTTPS (required for browser microphone access when deployed remotely):

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B --ssl-certfile cert.pem --ssl-keyfile key.pem --no-ssl-verify
```

### Streaming demo

```bash
qwen-asr-demo-streaming \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8000
```

Open `http://<your-ip>:8000` — the demo captures microphone audio and streams it to the model in real time.


## 🔧 Fine-tuning

See [finetuning/README.md](finetuning/README.md) for detailed instructions on fine-tuning Qwen3-ASR with JSONL audio-text pairs and multi-GPU training via `torchrun`.

```bash
# Setup
pip install -U qwen-asr datasets
pip install -U flash-attn --no-build-isolation  # recommended
```


## 🏗️ Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/qwen3_asr_introduction.png" width="90%"/>
</p>

The Qwen3-ASR family (0.6B and 1.7B) supports language identification and ASR for 52 languages and dialects. Both models leverage large-scale speech training data and the audio understanding capability of their foundation model, **Qwen3-Omni**. The 1.7B model achieves state-of-the-art performance among open-source ASR models and is competitive with the strongest proprietary APIs.

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg" width="100%"/>
</p>

**Model downloads:**

```bash
# HuggingFace
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B

# ModelScope (recommended for Mainland China)
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B  --local_dir ./Qwen3-ASR-0.6B
```


## 🌍 Supported Languages & Dialects

**30 languages:** Chinese (zh) · English (en) · Cantonese (yue) · Arabic (ar) · German (de) · French (fr) · Spanish (es) · Portuguese (pt) · Indonesian (id) · Italian (it) · Korean (ko) · Russian (ru) · Thai (th) · Vietnamese (vi) · Japanese (ja) · Turkish (tr) · Hindi (hi) · Malay (ms) · Dutch (nl) · Swedish (sv) · Danish (da) · Finnish (fi) · Polish (pl) · Czech (cs) · Filipino (fil) · Persian (fa) · Greek (el) · Hungarian (hu) · Macedonian (mk) · Romanian (ro)

**22 Chinese dialects:** Anhui · Dongbei · Fujian · Gansu · Guizhou · Hebei · Henan · Hubei · Hunan · Jiangxi · Ningxia · Shandong · Shaanxi · Shanxi · Sichuan · Tianjin · Yunnan · Zhejiang · Cantonese HK · Cantonese Guangdong · Wu · Minnan

The ForcedAligner supports: Chinese · English · Cantonese · French · German · Italian · Japanese · Korean · Portuguese · Russian · Spanish


## 🏎️ Local Performance Benchmarks

Measured on an NVIDIA RTX 3090 (24 GB) with `gpu_memory_utilization=0.8`:

| Model | Peak VRAM | RTF (long clip) | Real-time speedup |
|---|---|---|---|
| Qwen3-ASR-1.7B (FP16) | 22.3 GB | 0.0263 | ~38× |
| Qwen3-ASR-0.6B (FP16) | 22.1 GB | 0.0143 | **~70×** |
| Qwen3-ASR-0.6B (util=0.3, eager) | 14.1 GB | 0.0407 | ~24× |

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for detailed memory scaling analysis.


## Evaluation

During evaluation, we ran inference for all models with `dtype=torch.bfloat16` and set `max_new_tokens=1024` using vLLM. Greedy search was used for all decoding, and none of the tests specified a language parameter. The detailed evaluation results are shown below.

<details>
<summary>ASR Benchmarks on Public Datasets (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">English (en)</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Librispeech<br>clean | other</td>
      <td style="text-align: center;"><strong>1.39</strong> | 3.75</td>
      <td style="text-align: center;">2.89 | 3.56</td>
      <td style="text-align: center;">2.78 | 5.70</td>
      <td style="text-align: center;">1.51 | 3.97</td>
      <td style="text-align: center;">1.68 | 4.03</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">1.63 | <strong>3.38</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">GigaSpeech</td>
      <td style="text-align: center;">25.50</td>
      <td style="text-align: center;">9.37</td>
      <td style="text-align: center;">9.55</td>
      <td style="text-align: center;">9.76</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">8.88</td>
      <td style="text-align: center;"><strong>8.45</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-en</td>
      <td style="text-align: center;">9.08</td>
      <td style="text-align: center;">14.49</td>
      <td style="text-align: center;">13.78</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.92</td>
      <td style="text-align: center;"><strong>7.39</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-en</td>
      <td style="text-align: center;"><strong>2.40</strong></td>
      <td style="text-align: center;">2.94</td>
      <td style="text-align: center;">6.31</td>
      <td style="text-align: center;">4.08</td>
      <td style="text-align: center;">5.49</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">3.35</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">MLS-en</td>
      <td style="text-align: center;">5.12</td>
      <td style="text-align: center;"><strong>3.68</strong></td>
      <td style="text-align: center;">7.09</td>
      <td style="text-align: center;">4.87</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">6.00</td>
      <td style="text-align: center;">4.58</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Tedlium</td>
      <td style="text-align: center;">7.69</td>
      <td style="text-align: center;">6.15</td>
      <td style="text-align: center;">4.91</td>
      <td style="text-align: center;">6.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>3.85<strong></td>
      <td style="text-align: center;"><strong>4.50</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">VoxPopuli</td>
      <td style="text-align: center;">10.29</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">12.12</td>
      <td style="text-align: center;">12.05</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>9.96<strong></td>
      <td style="text-align: center;"><strong>9.15</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese (zh)</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech<br>net | meeting</td>
      <td style="text-align: center;">15.30 | 32.27</td>
      <td style="text-align: center;">14.43 | 13.47</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">9.86 | 19.11</td>
      <td style="text-align: center;">6.35 | -</td>
      <td style="text-align: center;">5.97 | 6.88</td>
      <td style="text-align: center;"><strong>4.97</strong> | <strong>5.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">AISHELL-2-test</td>
      <td style="text-align: center;">4.24</td>
      <td style="text-align: center;">11.62</td>
      <td style="text-align: center;">2.85</td>
      <td style="text-align: center;">5.06</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.15</td>
      <td style="text-align: center;"><strong>2.71</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">SpeechIO</td>
      <td style="text-align: center;">12.86</td>
      <td style="text-align: center;">5.30</td>
      <td style="text-align: center;">2.93</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.44</td>
      <td style="text-align: center;"><strong>2.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-zh</td>
      <td style="text-align: center;">2.44</td>
      <td style="text-align: center;">2.71</td>
      <td style="text-align: center;">2.69</td>
      <td style="text-align: center;">4.09</td>
      <td style="text-align: center;">3.51</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;"><strong>2.41</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.70</td>
      <td style="text-align: center;">5.95</td>
      <td style="text-align: center;">12.91</td>
      <td style="text-align: center;">6.20</td>
      <td style="text-align: center;">6.89</td>
      <td style="text-align: center;"><strong>5.35</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Dialect</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">KeSpeech</td>
      <td style="text-align: center;">26.87</td>
      <td style="text-align: center;">24.71</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">28.79</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">7.08</td>
      <td style="text-align: center;"><strong>5.10</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-yue</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.43</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.18</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.79</td>
      <td style="text-align: center;"><strong>3.98</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-yue</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">18.76</td>
      <td style="text-align: center;">13.20</td>
      <td style="text-align: center;">16.23</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">9.50</td>
      <td style="text-align: center;"><strong>7.57</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh-tw</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.31</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;">7.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.59</td>
      <td style="text-align: center;"><strong>3.77</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Yue<br>short | long</td>
      <td style="text-align: center;">15.62 | 25.29</td>
      <td style="text-align: center;">25.19 | 11.23</td>
      <td style="text-align: center;">9.74 | 11.40</td>
      <td style="text-align: center;">32.26 | 46.64</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">7.54 | 9.92</td>
      <td style="text-align: center;"><strong>5.82</strong> | <strong>8.85</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Chuan<br>easy | hard</td>
      <td style="text-align: center;">34.81 | 53.98</td>
      <td style="text-align: center;">43.79 | 67.30</td>
      <td style="text-align: center;"><strong>11.40<strong> | <strong>20.20</strong></td>
      <td style="text-align: center;">14.35 | 26.80</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">13.92 | 24.45</td>
      <td style="text-align: center;">11.99 | 21.63</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR Benchmarks on Internal Datasets (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Accented English</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Accented English</td>
      <td style="text-align: center;">28.56</td>
      <td style="text-align: center;">23.85</td>
      <td style="text-align: center;">20.41</td>
      <td style="text-align: center;">21.30</td>
      <td style="text-align: center;">19.96</td>
      <td style="text-align: center;"><strong>16.62<strong></td>
      <td style="text-align: center;"><strong>16.07</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Mandarin</td>
    </tr>
    <tr>
      <td style="text-align: left;">Elders&Kids</td>
      <td style="text-align: center;">14.27</td>
      <td style="text-align: center;">36.93</td>
      <td style="text-align: center;">4.17</td>
      <td style="text-align: center;">10.61</td>
      <td style="text-align: center;">4.54</td>
      <td style="text-align: center;">4.48</td>
      <td style="text-align: center;"><strong>3.81</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">ExtremeNoise</td>
      <td style="text-align: center;">36.11</td>
      <td style="text-align: center;">29.06</td>
      <td style="text-align: center;">17.04</td>
      <td style="text-align: center;">63.17</td>
      <td style="text-align: center;">36.55</td>
      <td style="text-align: center;">17.88</td>
      <td style="text-align: center;"><strong>16.17</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">TongueTwister</td>
      <td style="text-align: center;">20.87</td>
      <td style="text-align: center;">4.97</td>
      <td style="text-align: center;">3.47</td>
      <td style="text-align: center;">16.63</td>
      <td style="text-align: center;">9.02</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;"><strong>2.44</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Mandarin</td>
      <td style="text-align: center;">20.73</td>
      <td style="text-align: center;">12.50</td>
      <td style="text-align: center;">6.61</td>
      <td style="text-align: center;">14.01</td>
      <td style="text-align: center;">7.32</td>
      <td style="text-align: center;">7.06</td>
      <td style="text-align: center;"><strong>6.54</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Dialect</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Cantonese</td>
      <td style="text-align: center;">16.05</td>
      <td style="text-align: center;">14.98</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">31.04</td>
      <td style="text-align: center;">5.85</td>
      <td style="text-align: center;"><strong>4.80<strong></td>
      <td style="text-align: center;"><strong>4.12</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Chinese Dialects</td>
      <td style="text-align: center;">45.37</td>
      <td style="text-align: center;">47.70</td>
      <td style="text-align: center;">19.85</td>
      <td style="text-align: center;">44.55</td>
      <td style="text-align: center;">19.41</td>
      <td style="text-align: center;"><strong>18.24<strong></td>
      <td style="text-align: center;"><strong>15.94</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Dialect coverage:</strong> Results for <em>Dialog-Accented English</em> are averaged over 16 accents, and results for <em>Dialog-Chinese Dialects</em> are averaged over 22 Chinese dialects.</p>

</details>

<details>
<summary>Multilingual ASR Benchmarks (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GLM-ASR<br>-Nano-2512</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Open-sourced Benchmarks</td>
    </tr>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;">13.32</td>
      <td style="text-align: center;">8.62</td>
      <td style="text-align: center;">28.70</td>
      <td style="text-align: center;">13.19</td>
      <td style="text-align: center;"><strong>8.55</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">19.40</td>
      <td style="text-align: center;">10.77</td>
      <td style="text-align: center;">17.25</td>
      <td style="text-align: center;">12.75</td>
      <td style="text-align: center;"><strong>9.18</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">34.93</td>
      <td style="text-align: center;">15.68</td>
      <td style="text-align: center;">29.94</td>
      <td style="text-align: center;">15.84</td>
      <td style="text-align: center;"><strong>12.74</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">16.08</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">10.03</td>
      <td style="text-align: center;">7.57</td>
      <td style="text-align: center;"><strong>4.90</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>†</sup></td>
      <td style="text-align: center;">20.05</td>
      <td style="text-align: center;">6.85</td>
      <td style="text-align: center;">31.89</td>
      <td style="text-align: center;">10.37</td>
      <td style="text-align: center;"><strong>6.62</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>††</sup></td>
      <td style="text-align: center;">24.83</td>
      <td style="text-align: center;"><strong>8.16</strong></td>
      <td style="text-align: center;">47.84</td>
      <td style="text-align: center;">21.80</td>
      <td style="text-align: center;">12.60</td>
    </tr>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Qwen-ASR Internal Benchmarks</td>
    </tr>
    <tr>
      <td style="text-align: left;">News-Multilingual</td>
      <td style="text-align: center;">49.40</td>
      <td style="text-align: center;">14.80</td>
      <td style="text-align: center;">65.07</td>
      <td style="text-align: center;">17.39</td>
      <td style="text-align: center;"><strong>12.80</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Language coverage:</strong> <em>MLS</em> includes 8 languages: {da, de, en, es, fr, it, pl, pt}.<br><em>CommonVoice</em> includes 13 languages: {en, zh, yue, zh_TW, ar, de, es, fr, it, ja, ko, pt, ru}.<br><em>MLC-SLM</em> includes 11 languages: {en, fr, de, it, pt, es, ja, ko, ru, th, vi}.<br><em>Fleurs</em> includes 12 languages: {en, zh, yue, ar, de, es, fr, it, ja, ko, pt, ru }.<br><em>Fleurs<sup>†</sup></em> includes 8 additional languages beyond Fleurs: {hi, id, ms, nl, pl, th, tr, vi}.<br><em>Fleurs<sup>††</sup></em> includes 10 additional languages beyond Fleurs<sup>†</sup>: {cs, da, el, fa, fi, fil, hu, mk, ro, sv}.<br><em>News-Multilingual</em> includes 15 languages: {ar, de, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, vi}.</p>

</details>

<details>
<summary>Language Identification Accuracy (%) ↑</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Whisper-large-v3</th>
      <th style="text-align: center;">Qwen3-ASR-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
      <td style="text-align: center;">99.3</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">92.7</td>
      <td style="text-align: center;"><strong>98.2<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">89.2</td>
      <td style="text-align: center;"><strong>92.7<strong></td>
      <td style="text-align: center;"><strong>94.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">94.6</td>
      <td style="text-align: center;"><strong>97.1<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">94.1</td>
      <td style="text-align: center;"><strong>96.8<strong></td>
      <td style="text-align: center;"><strong>97.9</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Language coverage:</strong> The language sets follow Multilingual ASR Benchmarks. Here, Fleurs corresponds to Fleurs<sup>††</sup> in Multilingual ASR Benchmarks and covers 30 languages.</p>

</details>

<details>
<summary>Singing Voice & Song Transcription (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR<br>-1.0</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR-MLT<br>-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Singing</td>
    </tr>
    <tr>
      <td style="text-align: left;">M4Singer</td>
      <td style="text-align: center;">16.77</td>
      <td style="text-align: center;">20.88</td>
      <td style="text-align: center;">7.88</td>
      <td style="text-align: center;">13.58</td>
      <td style="text-align: center;">7.29</td>
      <td style="text-align: center;"><strong>5.98</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MIR-1k-vocal</td>
      <td style="text-align: center;">11.87</td>
      <td style="text-align: center;">9.85</td>
      <td style="text-align: center;">6.56</td>
      <td style="text-align: center;">11.71</td>
      <td style="text-align: center;">8.17</td>
      <td style="text-align: center;"><strong>6.25</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Opencpop</td>
      <td style="text-align: center;">7.93</td>
      <td style="text-align: center;">6.49</td>
      <td style="text-align: center;">3.80</td>
      <td style="text-align: center;">9.52</td>
      <td style="text-align: center;"><strong>2.98</strong></td>
      <td style="text-align: center;">3.08</td>
    </tr>
    <tr>
      <td style="text-align: left;">Popcs</td>
      <td style="text-align: center;">32.84</td>
      <td style="text-align: center;">15.13</td>
      <td style="text-align: center;">8.97</td>
      <td style="text-align: center;">13.77</td>
      <td style="text-align: center;">9.42</td>
      <td style="text-align: center;"><strong>8.52</strong></td>
    </tr>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Songs with BGM</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-en</td>
      <td style="text-align: center;">30.71</td>
      <td style="text-align: center;"><strong>12.18</strong></td>
      <td style="text-align: center;">33.51</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">14.60</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-zh</td>
      <td style="text-align: center;">34.86</td>
      <td style="text-align: center;">18.68</td>
      <td style="text-align: center;">23.99</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;"><strong>13.91</strong></td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR Inference Mode Performance (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: left;">Infer. Mode</th>
      <th style="text-align: center;">Librispeech</th>
      <th style="text-align: center;">Fleurs-en</th>
      <th style="text-align: center;">Fleurs-zh</th>
      <th style="text-align: center;">Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-1.7B</td>
      <td style="text-align: left;">Offline</td>
      <td style="text-align: center;">1.63 | 3.38</td>
      <td style="text-align: center;">3.35</td>
      <td style="text-align: center;">2.41</td>
      <td style="text-align: center;">2.69</td>
    </tr>
    <tr>
      <td style="text-align: left;">Streaming</td>
      <td style="text-align: center;">1.95 | 4.51</td>
      <td style="text-align: center;">4.02</td>
      <td style="text-align: center;">2.84</td>
      <td style="text-align: center;">3.33</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-0.6B</td>
      <td style="text-align: left;">Offline</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;">3.48</td>
    </tr>
    <tr>
      <td style="text-align: left;">Streaming</td>
      <td style="text-align: center;">2.54 | 6.27</td>
      <td style="text-align: center;">5.38</td>
      <td style="text-align: center;">3.40</td>
      <td style="text-align: center;">4.40</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>Forced Alignment Benchmarks (AAS ms ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Monotonic-Aligner</th>
      <th style="text-align: center;">NFA</th>
      <th style="text-align: center;">WhisperX</th>
      <th style="text-align: center;">Qwen3-ForcedAligner-0.6B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA-Labeled Raw</td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">109.8</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>33.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">107.5</td>
      <td style="text-align: center;">92.1</td>
      <td style="text-align: center;"><strong>37.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">100.7</td>
      <td style="text-align: center;">145.3</td>
      <td style="text-align: center;"><strong>41.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">122.7</td>
      <td style="text-align: center;">165.1</td>
      <td style="text-align: center;"><strong>46.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">142.7</td>
      <td style="text-align: center;">155.5</td>
      <td style="text-align: center;"><strong>75.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>37.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>38.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">200.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>40.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">124.7</td>
      <td style="text-align: center;">108.0</td>
      <td style="text-align: center;"><strong>36.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">129.8</td>
      <td style="text-align: center;">133.2</td>
      <td style="text-align: center;"><strong>42.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA-Labeled Concat-300s</td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">235.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>36.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">226.7</td>
      <td style="text-align: center;">227.2</td>
      <td style="text-align: center;"><strong>58.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">230.6</td>
      <td style="text-align: center;">2052.2</td>
      <td style="text-align: center;"><strong>53.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">220.3</td>
      <td style="text-align: center;">993.4</td>
      <td style="text-align: center;"><strong>62.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">290.5</td>
      <td style="text-align: center;">5719.4</td>
      <td style="text-align: center;"><strong>81.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>81.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>50.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">283.3</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>43.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">240.2</td>
      <td style="text-align: center;">4549.9</td>
      <td style="text-align: center;"><strong>39.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Cross-lingual</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>34.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">246.7</td>
      <td style="text-align: center;">2708.4</td>
      <td style="text-align: center;"><strong>52.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Human-Labeled</td>
    </tr>
    <tr>
      <td style="text-align: left;">Raw</td>
      <td style="text-align: center;">49.9</td>
      <td style="text-align: center;">88.6</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>27.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Raw-Noisy</td>
      <td style="text-align: center;">53.3</td>
      <td style="text-align: center;">89.5</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>41.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-60s</td>
      <td style="text-align: center;">51.1</td>
      <td style="text-align: center;">86.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>25.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-300s</td>
      <td style="text-align: center;">410.8</td>
      <td style="text-align: center;">140.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>24.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-Cross-lingual</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">141.3</td>
      <td style="text-align: center;">101.2</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>32.4</strong></td>
    </tr>
  </tbody>
</table>

</details>


## 🐛 Troubleshooting

**Server won't start**
```bash
nvidia-smi                            # verify GPU is accessible
pip list | grep -E "fastapi|uvicorn|vllm"  # verify installation
```

**Out of memory errors**
```bash
export GPU_MEMORY_UTILIZATION="0.7"
export MAX_MODEL_LEN="4096"
export MODEL_ID="Qwen/Qwen3-ASR-0.6B"  # smaller model
```

**4-bit quantization not working**  
Expected with vLLM 0.14.0. The server automatically falls back to FP16/BF16. Check logs for the line `⚠ Model loaded in fallback mode`.

**Audio processing errors**
```bash
ffmpeg -version       # must be installed
apt-get install ffmpeg libsndfile1   # Ubuntu/Debian
brew install ffmpeg                  # macOS
```

**CUDA graph errors**
```bash
export ENFORCE_EAGER="true"   # disable CUDA graphs
```

**vLLM multiprocessing error**  
Wrap vLLM code in `if __name__ == "__main__":` — see [vLLM docs](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing).


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```


## License

This project is licensed under the [Apache 2.0 License](LICENSE).

[![Star History Chart](https://api.star-history.com/svg?repos=groxaxo/Qwen3-ASR-FastAPI&type=Date)](https://star-history.com/#groxaxo/Qwen3-ASR-FastAPI&Date)
