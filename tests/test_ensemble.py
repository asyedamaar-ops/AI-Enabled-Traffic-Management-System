"""Tests for the ensemble fusion engine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import patch
import datetime

from detection.ensemble import EnsembleEngine


ENSEMBLE_CFG = {
    "emergency_threshold": 0.85,
    "night_start_hour": 18,
    "night_end_hour": 6,
    "weights": {
        "day":   {"video": 0.60, "audio": 0.40},
        "night": {"video": 0.35, "audio": 0.65},
        "adverse_weather": {"video": 0.30, "audio": 0.70},
    },
}


def _engine():
    return EnsembleEngine(ENSEMBLE_CFG)


def test_no_emergency_below_threshold():
    engine = _engine()
    score, is_ev = engine.evaluate(0.2, 0.1)
    assert not is_ev
    assert 0.0 < score < 0.85


def test_emergency_detected_above_threshold():
    engine = _engine()
    # Force high scores from both modalities
    score, is_ev = engine.evaluate(0.95, 0.95)
    assert is_ev
    assert score >= 0.85


def test_combined_score_in_range():
    engine = _engine()
    for _ in range(50):
        import random
        v = random.random()
        a = random.random()
        score, _ = engine.evaluate(v, a)
        assert 0.0 <= score <= 1.0


@patch("detection.ensemble.datetime")
def test_night_weights_applied(mock_dt):
    """Weights should shift toward audio at night."""
    mock_dt.datetime.now.return_value = datetime.datetime(2025, 1, 1, 22, 0, 0)
    engine = _engine()
    w_video, w_audio = engine._get_weights()
    assert w_audio > w_video  # night: audio gets more weight


def test_adverse_weather_flag():
    engine = _engine()
    engine.adverse_weather = True
    w_video, w_audio = engine._get_weights()
    assert w_audio > w_video
