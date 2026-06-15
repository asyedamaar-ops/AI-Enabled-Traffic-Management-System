"""
Signal Controller
=================
Manages the current state of all traffic signals at an intersection.

Normal mode: uses the optimizer's density-based timings.
Emergency mode: overrides all signals to create a green corridor for
                the detected emergency vehicle lane, holding all others red.

State machine per lane:
    GREEN -> YELLOW -> RED -> (next lane GREEN) -> ...
"""

import logging
import time
from typing import Dict, Optional

from signaling.optimizer import SignalOptimizer

logger = logging.getLogger("traffic_mgmt.signal_controller")


class SignalState:
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class SignalController:
    """Drives traffic signals for a single intersection."""

    def __init__(self, signaling_cfg: dict, emergency_cfg: dict) -> None:
        self._cfg = signaling_cfg
        self._emergency_cfg = emergency_cfg
        self._optimizer = SignalOptimizer(signaling_cfg)

        self._yellow_duration: int = signaling_cfg.get("yellow_duration", 3)
        self._corridor_duration: int = emergency_cfg.get("green_corridor_duration", 30)
        self._cooldown: int = emergency_cfg.get("cooldown_seconds", 10)

        # Current state per lane: {lane_id: SignalState}
        self._states: Dict[int, str] = {}
        # Scheduled green durations: {lane_id: seconds}
        self._green_timings: Dict[int, int] = {}

        self._emergency_active: bool = False
        self._emergency_end_time: float = 0.0
        self._last_emergency_time: float = 0.0

        logger.info("Signal controller initialised.")

    # -- Public API -----------------------------------------------------------

    def update(self, lane_counts: Dict[int, int], is_emergency: bool) -> None:
        """
        Called every cycle by the main loop.

        Args:
            lane_counts: Vehicle counts per lane from the video detector.
            is_emergency: Whether the ensemble detected an emergency vehicle.
        """
        now = time.time()

        # Handle emergency override
        if is_emergency and not self._emergency_active:
            cooldown_elapsed = (now - self._last_emergency_time) >= self._cooldown
            if cooldown_elapsed:
                self._activate_emergency(lane_counts, now)

        # Check if emergency corridor has expired
        if self._emergency_active and now >= self._emergency_end_time:
            self._deactivate_emergency()

        # Normal dynamic signaling
        if not self._emergency_active:
            self._green_timings = self._optimizer.compute(lane_counts)
            self._states = {
                lane: SignalState.GREEN if t == max(self._green_timings.values()) else SignalState.RED
                for lane, t in self._green_timings.items()
            }

    def get_states(self) -> Dict[int, str]:
        """Return the current signal state for each lane."""
        return dict(self._states)

    def get_timings(self) -> Dict[int, int]:
        """Return the current green-time allocations."""
        return dict(self._green_timings)

    def is_emergency_active(self) -> bool:
        return self._emergency_active

    # -- Internal -------------------------------------------------------------

    def _activate_emergency(self, lane_counts: Dict[int, int], now: float) -> None:
        """Override signals to clear a path for the emergency vehicle."""
        self._emergency_active = True
        self._emergency_end_time = now + self._corridor_duration
        self._last_emergency_time = now

        # Give green to the busiest lane (most likely where the EV is approaching)
        if lane_counts:
            priority_lane = max(lane_counts, key=lane_counts.get)
        else:
            priority_lane = 0

        self._states = {
            lane: (SignalState.GREEN if lane == priority_lane else SignalState.RED)
            for lane in (lane_counts.keys() or range(4))
        }
        self._green_timings = {lane: self._corridor_duration for lane in self._states}

        logger.warning(
            f"EMERGENCY OVERRIDE ACTIVE — Green corridor on lane {priority_lane} "
            f"for {self._corridor_duration}s."
        )

    def _deactivate_emergency(self) -> None:
        self._emergency_active = False
        logger.info("Emergency override expired. Resuming normal signaling.")
