"""
Signal Timing Optimizer
=======================
Converts per-lane vehicle counts into optimal green-light durations.

The algorithm is deliberately simple and explainable:
  - Each lane gets a share of the available cycle time proportional
    to its vehicle count relative to the total.
  - Hard bounds (min/max green) prevent starvation of empty lanes
    and stop busy lanes from monopolising the intersection forever.

This is a solid baseline. Swap in a reinforcement-learning policy
here if you want to experiment with smarter control.
"""

import logging
from typing import Dict

logger = logging.getLogger("traffic_mgmt.optimizer")


class SignalOptimizer:
    """Maps {lane_id: vehicle_count} -> {lane_id: green_seconds}."""

    def __init__(self, signaling_cfg: dict) -> None:
        self._min_green: int = signaling_cfg.get("min_green_seconds", 10)
        self._max_green: int = signaling_cfg.get("max_green_seconds", 90)
        self._base_green: int = signaling_cfg.get("base_green_seconds", 30)
        self._max_vehicles: int = 50  # normalisation cap

    def compute(self, lane_counts: Dict[int, int]) -> Dict[int, int]:
        """
        Args:
            lane_counts: {lane_id: vehicle_count_in_frame}

        Returns:
            {lane_id: recommended_green_seconds}
        """
        if not lane_counts:
            return {}

        total = sum(lane_counts.values())
        timings: Dict[int, int] = {}

        for lane_id, count in lane_counts.items():
            if total == 0:
                green = self._base_green
            else:
                # Proportional share of a fixed budget, scaled to [min, max]
                share = count / total
                green = self._min_green + share * (self._max_green - self._min_green)

            green = int(max(self._min_green, min(self._max_green, green)))
            timings[lane_id] = green

        logger.debug(f"Optimised timings: {timings} (counts: {lane_counts})")
        return timings
