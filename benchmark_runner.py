import time
import requests
import librosa
import os
import subprocess
import json
import threading


def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits"
        output = (
            subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")
        )
        # We assume we care about the GPU currently in use (e.g., set by CUDA_VISIBLE_DEVICES)
        # However, for simplicity, we'll just log all and we can look at the active one.
        return output
    except Exception as e:
        return [f"Error: {e}"]


class ResourceMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.samples.append((time.time(), get_gpu_stats()))
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


def measure_benchmark(audio_path, api_url, label):
    print(f"[*] Benchmarking {label} on {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    start_time = time.time()
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            data = {"model": "Qwen/Qwen3-ASR-1.7B"}
            response = requests.post(api_url, files=files, data=data)
    except Exception as e:
        print(f"[!] Request failed: {e}")
        monitor.stop()
        return None

    end_time = time.time()
    processing_time = end_time - start_time
    monitor.stop()
    monitor.join()

    if response.status_code == 200:
        rtf = processing_time / duration

        # Analyze samples (assuming GPU 1 based on previous setup)
        # Sample format: 'mem_used, gpu_util'
        gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        mem_samples = []
        util_samples = []
        for _, stats in monitor.samples:
            if len(stats) > gpu_index:
                data_point = stats[gpu_index].split(",")
                mem_samples.append(float(data_point[0]))
                util_samples.append(float(data_point[1]))

        avg_mem = sum(mem_samples) / len(mem_samples) if mem_samples else 0
        max_mem = max(mem_samples) if mem_samples else 0
        avg_util = sum(util_samples) / len(util_samples) if util_samples else 0

        results = {
            "label": label,
            "filename": audio_path,
            "duration": duration,
            "processing_time": processing_time,
            "rtf": rtf,
            "avg_vram_mb": avg_mem,
            "max_vram_mb": max_mem,
            "avg_gpu_util": avg_util,
        }
        print(f"[+] {label} Result: RTF={rtf:.4f}, MaxVRAM={max_mem:.1f}MB")
        return results
    else:
        print(f"[!] Transcription failed ({response.status_code})")
        return None


if __name__ == "__main__":
    import sys

    # For standalone use
    audio_files = ["bench_1_short.wav", "bench_2_med.wav", "bench_3_long.wav"]
    server_url = "http://localhost:8001/v1/audio/transcriptions"
    mode_label = sys.argv[1] if len(sys.argv) > 1 else "Unknown"

    all_results = []
    for f in audio_files:
        if os.path.exists(f):
            res = measure_benchmark(f, server_url, mode_label)
            if res:
                all_results.append(res)

    with open(f"results_{mode_label.lower()}.json", "w") as out:
        json.dump(all_results, out, indent=2)
