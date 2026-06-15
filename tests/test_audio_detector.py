"""Tests for audio detector — runs without microphone hardware."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np


AUDIO_CFG = {
    "model_path": "models/cnn_siren.h5",
    "clip_duration": 3,
    "overlap": 1,
    "n_mfcc": 40,
    "n_fft": 2048,
    "hop_length": 512,
    "confidence_threshold": 0.60,
}
CAPTURE_CFG = {
    "sample_rate": 44100,
    "channels": 1,
    "chunk_size": 1024,
    "device_index": None,
}


def test_init_no_hardware():
    """AudioDetector should fall back to mock mode gracefully."""
    from detection.audio_detector import AudioDetector
    det = AudioDetector(AUDIO_CFG, CAPTURE_CFG)
    assert det._stream is None  # no real mic in test env


def test_get_latest_returns_float():
    from detection.audio_detector import AudioDetector
    det = AudioDetector(AUDIO_CFG, CAPTURE_CFG)
    score = det.get_latest()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
