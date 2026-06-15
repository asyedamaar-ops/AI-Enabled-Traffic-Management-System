"""Tests for the signal controller and optimizer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from signaling.optimizer import SignalOptimizer
from signaling.signal_controller import SignalController, SignalState


SIGNAL_CFG = {
    "min_green_seconds": 10,
    "max_green_seconds": 90,
    "base_green_seconds": 30,
    "yellow_duration": 3,
    "cycle_interval": 1.0,
}
EMERGENCY_CFG = {
    "green_corridor_duration": 30,
    "cooldown_seconds": 0,  # disable cooldown for tests
    "alert_sound": False,
}


def test_optimizer_proportional_timing():
    opt = SignalOptimizer(SIGNAL_CFG)
    counts = {0: 10, 1: 5, 2: 0, 3: 15}
    timings = opt.compute(counts)

    assert all(SIGNAL_CFG["min_green_seconds"] <= t <= SIGNAL_CFG["max_green_seconds"]
               for t in timings.values())
    # Busiest lane should get more time than emptiest
    assert timings[3] >= timings[2]


def test_optimizer_empty_input():
    opt = SignalOptimizer(SIGNAL_CFG)
    result = opt.compute({})
    assert result == {}


def test_optimizer_all_zero_counts():
    opt = SignalOptimizer(SIGNAL_CFG)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    timings = opt.compute(counts)
    assert all(t == SIGNAL_CFG["base_green_seconds"] for t in timings.values())


def test_controller_normal_update():
    ctrl = SignalController(SIGNAL_CFG, EMERGENCY_CFG)
    ctrl.update({0: 5, 1: 10, 2: 3, 3: 8}, is_emergency=False)
    states = ctrl.get_states()
    assert any(s == SignalState.GREEN for s in states.values())
    assert not ctrl.is_emergency_active()


def test_controller_emergency_override():
    ctrl = SignalController(SIGNAL_CFG, EMERGENCY_CFG)
    ctrl.update({0: 5, 1: 20, 2: 3, 3: 8}, is_emergency=True)
    assert ctrl.is_emergency_active()
    states = ctrl.get_states()
    green_lanes = [lane for lane, s in states.items() if s == SignalState.GREEN]
    assert len(green_lanes) == 1  # exactly one lane gets the green corridor
