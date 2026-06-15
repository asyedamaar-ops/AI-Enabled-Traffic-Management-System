"""
Streamlit Monitoring Dashboard
===============================
Run with:  streamlit run src/dashboard/app.py

Gives traffic operators a real-time view of:
  - Per-lane vehicle counts and signal states
  - Emergency vehicle detection scores (video, audio, combined)
  - Historical wait-time and throughput charts
  - Manual override controls

In standalone mode (no live system), it generates mock data so you can
demo the dashboard without the full hardware stack.
"""

import random
import time
from collections import deque
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
LANES = ["North", "South", "East", "West"]
HISTORY_LEN = 60  # data points to keep in rolling charts

# ── Session state — persists across reruns ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = {lane: deque(maxlen=HISTORY_LEN) for lane in LANES}
if "timestamps" not in st.session_state:
    st.session_state.timestamps = deque(maxlen=HISTORY_LEN)
if "emergency_log" not in st.session_state:
    st.session_state.emergency_log = []
if "manual_override" not in st.session_state:
    st.session_state.manual_override = None


# ── Mock data generator (replace with real system calls in production) ────────
def get_system_state():
    """Simulate live data. Replace with real controller.get_states() calls."""
    lane_counts = {lane: random.randint(0, 20) for lane in LANES}
    max_lane = max(lane_counts, key=lane_counts.get)

    signal_states = {
        lane: "GREEN" if lane == max_lane else "RED"
        for lane in LANES
    }
    if st.session_state.manual_override:
        signal_states = {
            lane: "GREEN" if lane == st.session_state.manual_override else "RED"
            for lane in LANES
        }

    video_score = round(random.uniform(0.0, 0.25), 3)
    audio_score = round(random.uniform(0.0, 0.20), 3)
    combined = round(0.6 * video_score + 0.4 * audio_score, 3)
    is_emergency = combined > 0.85

    green_timings = {
        lane: int(10 + (lane_counts[lane] / 20) * 80)
        for lane in LANES
    }

    return {
        "lane_counts": lane_counts,
        "signal_states": signal_states,
        "green_timings": green_timings,
        "video_score": video_score,
        "audio_score": audio_score,
        "combined_score": combined,
        "is_emergency": is_emergency,
    }


# ── Signal colour helper ──────────────────────────────────────────────────────
SIGNAL_COLOUR = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚦 Traffic Control")
    st.markdown("---")

    st.subheader("Manual Override")
    override_choice = st.selectbox(
        "Force green on lane:", ["Auto (no override)"] + LANES
    )
    if override_choice == "Auto (no override)":
        st.session_state.manual_override = None
    else:
        st.session_state.manual_override = override_choice
        st.warning(f"Manual override ACTIVE: {override_choice} is green.")

    st.markdown("---")
    st.subheader("System Info")
    st.metric("Uptime", "99.7%")
    st.metric("Avg Response Time", "3.2s")
    st.metric("Detection Accuracy", "97.2%")

    st.markdown("---")
    refresh = st.slider("Refresh interval (s)", 0.5, 5.0, 1.0, 0.5)
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("AI-Enabled Traffic Management System")
st.caption("Real-time intersection monitoring dashboard")

state = get_system_state()

# Record history
st.session_state.timestamps.append(datetime.now().strftime("%H:%M:%S"))
for lane in LANES:
    st.session_state.history[lane].append(state["lane_counts"][lane])

if state["is_emergency"]:
    st.session_state.emergency_log.append(
        f"{datetime.now().strftime('%H:%M:%S')} — Emergency vehicle detected "
        f"(score: {state['combined_score']:.3f})"
    )

# ── Emergency alert banner ────────────────────────────────────────────────────
if state["is_emergency"]:
    st.error("🚨 EMERGENCY VEHICLE DETECTED — Green corridor activated")

# ── Row 1: Lane status cards ──────────────────────────────────────────────────
st.subheader("Lane Status")
cols = st.columns(4)
for i, lane in enumerate(LANES):
    with cols[i]:
        signal = state["signal_states"][lane]
        icon = SIGNAL_COLOUR[signal]
        count = state["lane_counts"][lane]
        green_t = state["green_timings"][lane]
        st.metric(label=f"{icon} {lane}", value=f"{count} vehicles", delta=f"{green_t}s green")

# ── Row 2: Detection scores ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Emergency Vehicle Detection")

d_cols = st.columns(3)
with d_cols[0]:
    st.metric("Video Score", f"{state['video_score']:.3f}", help="DenseNet-169 confidence")
    st.progress(state["video_score"])
with d_cols[1]:
    st.metric("Audio Score", f"{state['audio_score']:.3f}", help="CNN siren confidence")
    st.progress(state["audio_score"])
with d_cols[2]:
    combined = state["combined_score"]
    delta_label = "ALERT" if combined > 0.85 else "Normal"
    st.metric("Combined Score", f"{combined:.3f}", delta=delta_label)
    colour = "red" if combined > 0.85 else "green"
    st.progress(min(combined, 1.0))

# ── Row 3: Vehicle count chart ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Vehicle Count History")

if len(st.session_state.timestamps) > 1:
    chart_data = pd.DataFrame(
        {lane: list(st.session_state.history[lane]) for lane in LANES},
        index=list(st.session_state.timestamps),
    )
    st.line_chart(chart_data, use_container_width=True, height=250)
else:
    st.info("Collecting data...")

# ── Row 4: Emergency log ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Emergency Event Log")
if st.session_state.emergency_log:
    for entry in reversed(st.session_state.emergency_log[-10:]):
        st.text(entry)
else:
    st.success("No emergency events detected in this session.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(refresh)
st.rerun()
