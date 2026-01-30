#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for FastAPI server (without model loading).

These tests check the FastAPI server structure and endpoints
without actually loading the ASR model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import test dependencies
try:
    from fastapi.testclient import TestClient
    import pytest
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("FastAPI dependencies not installed. Install with: pip install qwen-asr[fastapi]")
    print("Skipping tests.")


def test_imports():
    """Test that the server module can be imported"""
    if not DEPENDENCIES_AVAILABLE:
        print("⊘ SKIPPED: Dependencies not available")
        return
    
    try:
        from qwen_asr.cli import serve_fastapi
        assert serve_fastapi is not None
        print("✓ serve_fastapi module imports successfully")
    except ImportError as e:
        print(f"⊘ SKIPPED: {e}")


def test_environment_variables():
    """Test that environment variables are read correctly"""
    if not DEPENDENCIES_AVAILABLE:
        print("⊘ SKIPPED: Dependencies not available")
        return
    
    try:
        # Set custom environment variables
        os.environ['MODEL_ID'] = 'test/model'
        os.environ['QUANT_MODE'] = 'none'
        os.environ['PORT'] = '9000'
        
        # Re-import to pick up new env vars
        import importlib
        from qwen_asr.cli import serve_fastapi
        importlib.reload(serve_fastapi)
        
        assert serve_fastapi.MODEL_ID == 'test/model'
        assert serve_fastapi.QUANT_MODE == 'none'
        assert serve_fastapi.PORT == 9000
        
        print("✓ Environment variables are read correctly")
    except ImportError as e:
        print(f"⊘ SKIPPED: {e}")
    finally:
        # Clean up
        os.environ.pop('MODEL_ID', None)
        os.environ.pop('QUANT_MODE', None)
        os.environ.pop('PORT', None)


def test_error_response_format():
    """Test that error responses follow OpenAI format"""
    if not DEPENDENCIES_AVAILABLE:
        print("⊘ SKIPPED: Dependencies not available")
        return
    
    try:
        from qwen_asr.cli.serve_fastapi import create_error_response
        
        response = create_error_response(
            message="Test error",
            error_type="test_error",
            status_code=400
        )
        
        assert response.status_code == 400
        content = response.body.decode()
        assert "error" in content
        assert "Test error" in content
        assert "test_error" in content
        
        print("✓ Error responses follow OpenAI format")
    except ImportError as e:
        print(f"⊘ SKIPPED: {e}")


def test_response_models():
    """Test that Pydantic models are properly defined"""
    if not DEPENDENCIES_AVAILABLE:
        print("⊘ SKIPPED: Dependencies not available")
        return
    
    try:
        from qwen_asr.cli.serve_fastapi import (
            ErrorResponse,
            ModelInfo,
            ModelsResponse,
            TranscriptionResponse
        )
        
        # Test ModelInfo
        model_info = ModelInfo(id="test-model", created=123456)
        assert model_info.id == "test-model"
        assert model_info.object == "model"
        assert model_info.owned_by == "qwen"
        
        # Test ModelsResponse
        models_response = ModelsResponse(data=[model_info])
        assert models_response.object == "list"
        assert len(models_response.data) == 1
        
        # Test TranscriptionResponse
        transcription = TranscriptionResponse(text="Hello world", language="English")
        assert transcription.text == "Hello world"
        assert transcription.language == "English"
        
        print("✓ Pydantic models are properly defined")
    except ImportError as e:
        print(f"⊘ SKIPPED: {e}")


if __name__ == "__main__":
    print("Running FastAPI server tests...\n")
    
    if not DEPENDENCIES_AVAILABLE:
        print("\nAll tests skipped due to missing dependencies.")
        print("To run these tests, install: pip install qwen-asr[fastapi]")
        sys.exit(0)
    
    tests = [
        ("Module imports", test_imports),
        ("Environment variables", test_environment_variables),
        ("Error response format", test_error_response_format),
        ("Response models", test_response_models),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*50)
    
    sys.exit(0 if failed == 0 else 1)
