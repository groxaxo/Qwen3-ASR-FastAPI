# Qwen ASR Benchmark Report: 1.7B vs 0.6B & Memory Scaling

This report compares performance, resource usage, and memory efficiency for the Qwen3-ASR model family on an NVIDIA GeForce RTX 3090.

## 1. Why vLLM Pre-allocates VRAM
vLLM pre-allocates a large "KV Cache Pool" (controlled by `gpu_memory_utilization`) for several critical reasons:
- **PagedAttention**: Efficiently manages memory for variable-length requests by pooling contiguous memory.
- **Avoid Fragmentation**: By claiming memory at startup, it prevents PyTorch's allocator from causing fragmentation during sustained high-throughput inference.
- **Continuous Batching**: It ensures a deterministic amount of space is available to "pack" as many parallel requests as possible without OOM crashes during runtime.

## 2. Model Performance Comparison
Measurements taken with `gpu_memory_utilization=0.8` (Default).

| Model | Weight Size | Peak VRAM | Long Clip RTF | Real-Time Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3-1.7B** | ~3.8 GiB | 22.29 GiB | 0.0263 | ~38x |
| **Qwen3-0.6B** | ~1.5 GiB | 22.15 GiB | **0.0143** | **~70x** |

## 3. Memory Scaling (0.6B Model)
We tested the limits of the 0.6B model by reducing `gpu_memory_utilization`. To hit lower floors, we used `enforce_eager=True` (disabling CUDA graphs) and reduced `max_model_len` to 8192.

| Util Setting | Mode | Peak VRAM | Status | Resulting RTF (Long) |
| :--- | :--- | :--- | :--- | :--- |
| **0.8** | CUDA Graphs | ~22.1 GB | Success | 0.0143 |
| **0.3** | Eager | ~14.1 GB | Success | 0.0407 |
| **0.2** | Eager | < 4.8 GB | **FAILED** | N/A (Insufficient KV Cache) |

### Findings on Scaling:
1. **The Floor**: For the 0.6B model (FP16), the absolute memory floor is ~0.25 utilization (~6GB) on a 24GB card. Setting it to 0.2 (4.8GB) fails because the weights (1.5GB) + activation overhead + metadata leave no room for the KV Cache blocks.
2. **Performance Trade-off**: Switching to **Eager Mode** (required for low-memory setups) results in a **~2.8x slower** RTF compared to CUDA Graph mode (0.0407 vs 0.0143), but still maintains an impressive **~24x speedup** over real-time.

## Conclusion
- **For Maximum Speed**: Use the 0.6B model with default settings (0.8 util). It is nearly twice as fast as the 1.7B model.
- **For Multi-service Density**: The 0.6B model can be safely restricted to **0.3 utilization (~7GB)** at the cost of some latency, allowing you to run 2-3 instances on a single 3090.
- **Wait for Quantization**: 4-bit support in vLLM will eventually allow the 0.6B model to run comfortably on 4GB-8GB cards.
