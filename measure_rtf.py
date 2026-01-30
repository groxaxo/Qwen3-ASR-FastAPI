import time
import requests
import librosa
import os


def measure_rtf(audio_path, api_url):
    print(f"[*] Loading audio to measure duration: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    print(f"[+] Audio Duration: {duration:.2f} seconds")

    print(f"[*] Starting transcription speed test for {audio_path}...")
    start_time = time.time()

    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"model": "Qwen/Qwen3-ASR-1.7B"}
        response = requests.post(api_url, files=files, data=data)

    end_time = time.time()
    processing_time = end_time - start_time

    if response.status_code == 200:
        result = response.json()
        rtf = processing_time / duration
        print(f"\n--- SPEED TEST RESULTS ---")
        print(f"Audio Duration:   {duration:.2f}s")
        print(f"Processing Time:  {processing_time:.2f}s")
        print(f"Real-Time Factor: {rtf:.4f} (Lower is better)")
        print(f"Transcription:    {result.get('text', '')[:100]}...")
        print(f"---------------------------\n")
        return rtf
    else:
        print(
            f"[!] Transcription failed with status {response.status_code}: {response.text}"
        )
        return None


if __name__ == "__main__":
    audio_file = "speed_test.wav"
    server_url = "http://localhost:8001/v1/audio/transcriptions"

    if os.path.exists(audio_file):
        measure_rtf(audio_file, server_url)
    else:
        print(f"[!] {audio_file} not found. Ensure the download finished.")
