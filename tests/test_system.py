"""
Tests for VideoDetector

Run with:  pytest tests/ -v
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# Minimal config that matches the shape of default.yaml
_VIDEO_CFG = {
    "sources": [{"id": 0}],
    "resolution": [640, 480],
    "fps_capture": 30,
    "fps_process": 10,
    "ssd_model_path": "models/ssd_vehicle.pb",      # won't exist in test env
    "ev_model_path": "models/densenet169_ev.h5",    # won't exist in test env
    "confidence_threshold": 0.5,
    "roi_margin": 0.05,
    "max_vehicles_per_lane": 50,
}


class TestVideoDetectorMockMode(unittest.TestCase):
    """Tests that run without real cameras or model files (mock / demo mode)."""

    def setUp(self):
        # Patch cv2.VideoCapture so no camera is needed
        self._cap_patcher = patch("cv2.VideoCapture")
        mock_cap_cls = self._cap_patcher.start()
        mock_cap_cls.return_value.isOpened.return_value = False  # force mock mode

        from src.detection.video_detector import VideoDetector
        self.detector = VideoDetector(_VIDEO_CFG, demo=False)

    def tearDown(self):
        self.detector.release()
        self._cap_patcher.stop()

    def test_get_latest_returns_expected_types(self):
        """get_latest() should return (float, dict) immediately."""
        score, counts = self.detector.get_latest()
        self.assertIsInstance(score, float)
        self.assertIsInstance(counts, dict)

    def test_run_populates_state(self):
        """After running briefly, get_latest() should return non-empty data."""
        stop = threading.Event()
        t = threading.Thread(target=self.detector.run, args=(stop,), daemon=True)
        t.start()
        time.sleep(0.3)
        stop.set()
        t.join(timeout=2)

        score, counts = self.detector.get_latest()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(counts, dict)

    def test_score_bounds(self):
        """Mock EV scores should always be in [0, 1]."""
        stop = threading.Event()
        t = threading.Thread(target=self.detector.run, args=(stop,), daemon=True)
        t.start()
        time.sleep(0.5)
        stop.set()
        t.join(timeout=2)

        score, _ = self.detector.get_latest()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestSignalOptimizer(unittest.TestCase):
    """Unit tests for the proportional green-time algorithm."""

    def setUp(self):
        from src.signaling.optimizer import SignalOptimizer
        self.opt = SignalOptimizer(min_green=10, max_green=90, base_green=30)

    def test_empty_counts_returns_empty(self):
        result = self.opt.compute({})
        self.assertEqual(result, {})

    def test_all_zero_counts_returns_minimum(self):
        result = self.opt.compute({0: 0, 1: 0, 2: 0, 3: 0})
        for v in result.values():
            self.assertEqual(v, 10.0)

    def test_proportional_allocation(self):
        """Busier lane should get longer green time."""
        result = self.opt.compute({0: 20, 1: 5})
        self.assertGreater(result[0], result[1])

    def test_clamp_to_bounds(self):
        """No timing should exceed max or fall below min."""
        result = self.opt.compute({0: 1000, 1: 0})
        self.assertLessEqual(result[0], 90.0)
        self.assertGreaterEqual(result[1], 10.0)

    def test_single_lane(self):
        """Single-lane intersection should still return a valid timing."""
        result = self.opt.compute({0: 15})
        self.assertIn(0, result)
        self.assertGreaterEqual(result[0], 10.0)
        self.assertLessEqual(result[0], 90.0)


class TestEnsembleEngine(unittest.TestCase):
    """Unit tests for the ensemble fusion logic."""

    def setUp(self):
        from src.detection.ensemble import EnsembleEngine
        cfg = {
            "emergency_threshold": 0.85,
            "night_start_hour": 18,
            "night_end_hour": 6,
            "cooldown_seconds": 0,   # disable cooldown in tests
            "weights": {
                "day": {"video": 0.6, "audio": 0.4},
                "night": {"video": 0.35, "audio": 0.65},
                "adverse_weather": {"video": 0.3, "audio": 0.7},
            },
        }
        self.engine = EnsembleEngine(cfg)

    def test_both_high_scores_trigger_emergency(self):
        score, is_emg = self.engine.evaluate(0.95, 0.95)
        self.assertTrue(is_emg)
        self.assertGreater(score, 0.85)

    def test_both_low_scores_no_emergency(self):
        score, is_emg = self.engine.evaluate(0.1, 0.05)
        self.assertFalse(is_emg)
        self.assertLess(score, 0.85)

    def test_combined_score_in_range(self):
        score, _ = self.engine.evaluate(0.7, 0.8)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_adverse_weather_shifts_weight_to_audio(self):
        # Same inputs, adverse_weather=True should produce a higher combined
        # score when audio is high and video is low (audio gets more weight)
        score_normal, _ = self.engine.evaluate(0.3, 0.9, adverse_weather=False)
        score_adverse, _ = self.engine.evaluate(0.3, 0.9, adverse_weather=True)
        self.assertGreater(score_adverse, score_normal)


if __name__ == "__main__":
    unittest.main()
