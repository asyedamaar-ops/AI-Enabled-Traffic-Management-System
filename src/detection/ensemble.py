"""
Ensemble Engine
===============
Combines probability scores from the video and audio detectors using
time-aware weighted averaging, then applies a threshold to decide
whether an emergency vehicle is present.
"""

import datetime
import logging
from typing import Tuple

logger = logging.getLogger("traffic_mgmt.ensemble")


class EnsembleEngine:
    """Fuses video + audio detection scores into a single emergency decision."""

    def __init__(self, ensemble_cfg: dict) -> None:
        self._cfg = ensemble_cfg
        self._threshold: float = ensemble_cfg.get("emergency_threshold", 0.85)
        self._night_start: int = ensemble_cfg.get("night_start_hour", 18)
        self._night_end: int = ensemble_cfg.get("night_end_hour", 6)

        weights_cfg = ensemble_cfg.get("weights", {})
        self._weights_day = weights_cfg.get("day", {"video": 0.60, "audio": 0.40})
        self._weights_night = weights_cfg.get("night", {"video": 0.35, "audio": 0.65})
        self._weights_adverse = weights_cfg.get("adverse_weather", {"video": 0.30, "audio": 0.70})

        self.adverse_weather: bool = False

        logger.info(
            f"Ensemble ready — threshold={self._threshold}, "
            f"day weights={self._weights_day}, night weights={self._weights_night}"
        )

    def evaluate(self, video_score: float, audio_score: float) -> Tuple[float, bool]:
        """
        Combine scores and return (combined_score, is_emergency).

        Args:
            video_score: P(emergency vehicle) from DenseNet-169 [0-1].
            audio_score: P(siren present) from CNN audio model [0-1].

        Returns:
            combined_score: Weighted average [0-1].
            is_emergency: True if combined_score >= threshold.
        """
        w_video, w_audio = self._get_weights()
        combined = w_video * video_score + w_audio * audio_score
        is_emergency = combined >= self._threshold

        if is_emergency:
            logger.warning(
                f"EMERGENCY VEHICLE DETECTED — combined={combined:.3f} "
                f"(video={video_score:.3f} x{w_video}, audio={audio_score:.3f} x{w_audio})"
            )

        return combined, is_emergency

    def _get_weights(self) -> Tuple[float, float]:
        if self.adverse_weather:
            w = self._weights_adverse
        elif self._is_night():
            w = self._weights_night
        else:
            w = self._weights_day
        return w["video"], w["audio"]

    def _is_night(self) -> bool:
        hour = datetime.datetime.now().hour
        if self._night_start > self._night_end:
            return hour >= self._night_start or hour < self._night_end
        return self._night_start <= hour < self._night_end
