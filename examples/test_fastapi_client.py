#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Example client for testing the Qwen3-ASR FastAPI server.

This script demonstrates how to use the OpenAI-compatible endpoints.
"""

import argparse
import json
import requests
import sys


def test_health(base_url: str):
    """Test the health check endpoint"""
    print("Testing /healthz endpoint...")
    response = requests.get(f"{base_url}/healthz")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_list_models(base_url: str):
    """Test the models listing endpoint"""
    print("Testing /v1/models endpoint...")
    response = requests.get(f"{base_url}/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_transcription(base_url: str, audio_file: str, language: str = None, response_format: str = "json"):
    """Test the audio transcription endpoint"""
    print(f"Testing /v1/audio/transcriptions endpoint with {audio_file}...")
    
    files = {
        'file': open(audio_file, 'rb')
    }
    
    data = {
        'response_format': response_format
    }
    
    if language:
        data['language'] = language
    
    try:
        response = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            if response_format == "text":
                print(f"Transcription: {response.text}\n")
            else:
                print(f"Response: {json.dumps(response.json(), indent=2)}\n")
        else:
            print(f"Error: {response.text}\n")
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"Error: {e}\n")
        return False
    finally:
        files['file'].close()


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR FastAPI server")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the FastAPI server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file for transcription test"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language for transcription (optional)"
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="json",
        choices=["json", "text"],
        help="Response format (default: json)"
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check test"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip models listing test"
    )
    
    args = parser.parse_args()
    
    base_url = args.base_url.rstrip("/")
    
    results = []
    
    # Test health endpoint
    if not args.skip_health:
        results.append(("Health Check", test_health(base_url)))
    
    # Test models endpoint
    if not args.skip_models:
        results.append(("List Models", test_list_models(base_url)))
    
    # Test transcription endpoint
    if args.audio_file:
        results.append((
            "Transcription",
            test_transcription(base_url, args.audio_file, args.language, args.response_format)
        ))
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    # Exit with error if any test failed
    if any(not success for _, success in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
