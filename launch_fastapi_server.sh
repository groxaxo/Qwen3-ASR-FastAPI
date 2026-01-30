#!/bin/bash
# Qwen3-ASR FastAPI Server Launcher
# 
# This script provides an easy way to launch the FastAPI server with
# common configurations.

set -e

# Default values
DEFAULT_MODEL="Qwen/Qwen3-ASR-1.7B"
DEFAULT_QUANT="4bit"
DEFAULT_DTYPE="bfloat16"
DEFAULT_GPU_MEM="0.8"
DEFAULT_MAX_LEN="8192"
DEFAULT_MAX_AUDIO="1200"
DEFAULT_MAX_UPLOAD="100"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"

# Help message
show_help() {
    cat << EOF
Qwen3-ASR FastAPI Server Launcher

Usage: $0 [OPTIONS]

Options:
    -m, --model MODEL           Model ID (default: ${DEFAULT_MODEL})
                                Options: Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B
    
    -q, --quant MODE           Quantization mode (default: ${DEFAULT_QUANT})
                                Options: 4bit, awq, gptq, none
    
    -d, --dtype TYPE           Data type (default: ${DEFAULT_DTYPE})
                                Options: float16, bfloat16
    
    -g, --gpu-mem UTIL         GPU memory utilization (default: ${DEFAULT_GPU_MEM})
                                Range: 0.0 to 1.0
    
    -l, --max-len LENGTH       Maximum model length (default: ${DEFAULT_MAX_LEN})
    
    -a, --max-audio SECONDS    Maximum audio duration (default: ${DEFAULT_MAX_AUDIO})
    
    -u, --max-upload MB        Maximum upload size in MB (default: ${DEFAULT_MAX_UPLOAD})
    
    -H, --host HOST            Server host (default: ${DEFAULT_HOST})
    
    -p, --port PORT            Server port (default: ${DEFAULT_PORT})
    
    -h, --help                 Show this help message

Presets:
    --low-vram                 Optimized for low VRAM systems (< 8GB)
                                Uses 0.6B model, 4bit quant, max_len=4096
    
    --high-performance         Optimized for high-performance systems
                                Uses 1.7B model, no quant, bfloat16

Examples:
    # Start with defaults
    $0

    # Low VRAM configuration
    $0 --low-vram

    # Custom configuration
    $0 -m Qwen/Qwen3-ASR-0.6B -q none -p 9000

    # High performance with custom GPU memory
    $0 --high-performance -g 0.9

Environment:
    All options can also be set via environment variables:
    MODEL_ID, QUANT_MODE, DTYPE, GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN, MAX_AUDIO_SECONDS, MAX_UPLOAD_SIZE_MB,
    HOST, PORT

EOF
}

# Parse arguments
PRESET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            export MODEL_ID="$2"
            shift 2
            ;;
        -q|--quant)
            export QUANT_MODE="$2"
            shift 2
            ;;
        -d|--dtype)
            export DTYPE="$2"
            shift 2
            ;;
        -g|--gpu-mem)
            export GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -l|--max-len)
            export MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -a|--max-audio)
            export MAX_AUDIO_SECONDS="$2"
            shift 2
            ;;
        -u|--max-upload)
            export MAX_UPLOAD_SIZE_MB="$2"
            shift 2
            ;;
        -H|--host)
            export HOST="$2"
            shift 2
            ;;
        -p|--port)
            export PORT="$2"
            shift 2
            ;;
        --low-vram)
            PRESET="low-vram"
            shift
            ;;
        --high-performance)
            PRESET="high-performance"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Apply presets
if [[ "$PRESET" == "low-vram" ]]; then
    echo "Applying low-VRAM preset..."
    export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
    export QUANT_MODE="${QUANT_MODE:-4bit}"
    export DTYPE="${DTYPE:-bfloat16}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
elif [[ "$PRESET" == "high-performance" ]]; then
    echo "Applying high-performance preset..."
    export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"
    export QUANT_MODE="${QUANT_MODE:-none}"
    export DTYPE="${DTYPE:-bfloat16}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
fi

# Set defaults for unset variables
export MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL}"
export QUANT_MODE="${QUANT_MODE:-$DEFAULT_QUANT}"
export DTYPE="${DTYPE:-$DEFAULT_DTYPE}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEM}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_LEN}"
export MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-$DEFAULT_MAX_AUDIO}"
export MAX_UPLOAD_SIZE_MB="${MAX_UPLOAD_SIZE_MB:-$DEFAULT_MAX_UPLOAD}"
export HOST="${HOST:-$DEFAULT_HOST}"
export PORT="${PORT:-$DEFAULT_PORT}"

# Print configuration
echo "=========================================="
echo "Qwen3-ASR FastAPI Server"
echo "=========================================="
echo "Model:                $MODEL_ID"
echo "Quantization:         $QUANT_MODE"
echo "Data Type:            $DTYPE"
echo "GPU Memory Util:      $GPU_MEMORY_UTILIZATION"
echo "Max Model Length:     $MAX_MODEL_LEN"
echo "Max Audio Duration:   ${MAX_AUDIO_SECONDS}s"
echo "Max Upload Size:      ${MAX_UPLOAD_SIZE_MB}MB"
echo "Server:               http://${HOST}:${PORT}"
echo "=========================================="
echo ""

# Check if qwen-asr-serve-fastapi is available
if ! command -v qwen-asr-serve-fastapi &> /dev/null; then
    echo "ERROR: qwen-asr-serve-fastapi not found"
    echo "Please install with: pip install qwen-asr[vllm,fastapi]"
    exit 1
fi

# Launch the server
echo "Starting server..."
echo ""
exec qwen-asr-serve-fastapi
