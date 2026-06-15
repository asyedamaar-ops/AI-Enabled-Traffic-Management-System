"""Tests for video detector — runs without real cameras or model weights."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


VIDEO_CFG = {
    "sources": [{"id": 0}],
    "resolution": [1280, 720],
    "fps_process": 10,
    "confidence_threshold": 0.5,
    "ssd_model_path": "models/ssd_vehicle.pb",
    "ev_model_path": "models/densenet169_ev.h5",
}


@patch("cv2.VideoCapture")
def test_init_no_cameras(mock_cap_cls):
    """Detector should initialise cleanly even when cameras fail to open."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap_cls.return_value = mock_cap

    from detection.video_detector import VideoDetector
    det = VideoDetector(VIDEO_CFG, demo=False)
    assert det._caps == []


@patch("cv2.VideoCapture")
def test_get_latest_mock_mode(mock_cap_cls):
    """get_latest() should return valid types in mock mode."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap_cls.return_value = mock_cap

    from detection.video_detector import VideoDetector
    det = VideoDetector(VIDEO_CFG, demo=False)

    score, counts = det.get_latest()
    assert isinstance(score, float)
    assert isinstance(counts, dict)
    assert 0.0 <= score <= 1.0
