"""
Video Detector
==============
Handles two tasks from a single camera feed:

1. **Vehicle counting** — SSD model detects every vehicle per lane and
   returns a density count used by the signal optimizer.

2. **Emergency vehicle detection** — DenseNet-169 classifier checks each
   frame for ambulances / fire trucks / police vehicles and returns a
   probability score [0.0 – 1.0].

The detector runs in its own thread (started by main.py) and exposes
`get_latest()` so the main loop can poll results without blocking.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("traffic_mgmt.video_detector")

# Vehicle class IDs from the COCO dataset that we care about
_VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Rough heuristic: which COCO classes might be emergency vehicles
# (DenseNet-169 does the fine-grained classification; SSD just flags candidates)
_EMERGENCY_CANDIDATE_IDS = {5, 7}  # bus and truck sized — ambulances / fire trucks


class VideoDetector:
    """Wraps OpenCV video capture and TF/Keras inference in one tidy class."""

    def __init__(self, video_cfg: dict, demo: bool = False) -> None:
        self._cfg = video_cfg
        self._demo = demo

        self._lock = threading.Lock()
        self._latest_ev_score: float = 0.0
        self._latest_lane_counts: Dict[int, int] = {}

        self._caps: List[cv2.VideoCapture] = []
        self._ssd_net = None
        self._ev_model = None

        self._init_cameras()
        self._load_models()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_cameras(self) -> None:
        sources = self._cfg.get("sources", [{"id": 0}])
        if self._demo:
            demo_path = Path("assets/sample_traffic.mp4")
            if demo_path.exists():
                sources = [{"id": str(demo_path)}]
                logger.info(f"Demo mode: using {demo_path}")
            else:
                logger.warning("Demo video not found; falling back to webcam 0.")
                sources = [{"id": 0}]

        for src in sources:
            cap = cv2.VideoCapture(src["id"])
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.get("resolution", [1280, 720])[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.get("resolution", [1280, 720])[1])
                self._caps.append(cap)
                logger.info(f"Camera opened: source={src['id']}")
            else:
                logger.warning(f"Could not open camera source: {src['id']}")

        if not self._caps:
            logger.warning("No cameras available. Running in mock mode.")

    def _load_models(self) -> None:
        ssd_path = Path(self._cfg.get("ssd_model_path", "models/ssd_vehicle.pb"))
        ev_path = Path(self._cfg.get("ev_model_path", "models/densenet169_ev.h5"))

        if ssd_path.exists():
            try:
                self._ssd_net = cv2.dnn.readNetFromTensorflow(str(ssd_path))
                logger.info("SSD vehicle detection model loaded.")
            except Exception as exc:
                logger.warning(f"Could not load SSD model ({exc}). Using mock detections.")
        else:
            logger.warning(f"SSD model not found at {ssd_path}. Using mock detections.")

        if ev_path.exists():
            try:
                # Lazy import so the package installs without TF in test envs
                from tensorflow import keras  # type: ignore
                self._ev_model = keras.models.load_model(str(ev_path))
                logger.info("DenseNet-169 EV classifier loaded.")
            except Exception as exc:
                logger.warning(f"Could not load EV model ({exc}). Using mock scores.")
        else:
            logger.warning(f"EV model not found at {ev_path}. Using mock scores.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, shutdown_event: threading.Event) -> None:
        """Main processing loop — runs in a background thread."""
        fps_process = self._cfg.get("fps_process", 10)
        interval = 1.0 / fps_process

        while not shutdown_event.is_set():
            tick = time.time()
            lane_counts: Dict[int, int] = {}
            ev_scores: List[float] = []

            for lane_id, cap in enumerate(self._caps):
                ret, frame = cap.read()
                if not ret:
                    continue

                count = self._count_vehicles(frame)
                ev_score = self._score_emergency_vehicle(frame)

                lane_counts[lane_id] = count
                ev_scores.append(ev_score)

            # If no real cameras, produce mock data so the rest of the system works
            if not self._caps:
                lane_counts = {i: np.random.randint(0, 15) for i in range(4)}
                ev_scores = [float(np.random.uniform(0, 0.2))]

            with self._lock:
                self._latest_lane_counts = lane_counts
                self._latest_ev_score = max(ev_scores) if ev_scores else 0.0

            elapsed = time.time() - tick
            sleep_for = max(0.0, interval - elapsed)
            shutdown_event.wait(timeout=sleep_for)

    def get_latest(self) -> Tuple[float, Dict[int, int]]:
        """Return (ev_probability_score, {lane_id: vehicle_count})."""
        with self._lock:
            return self._latest_ev_score, dict(self._latest_lane_counts)

    def release(self) -> None:
        """Release all camera handles."""
        for cap in self._caps:
            cap.release()
        logger.info("Video detector released.")

    # ── Internal Processing ───────────────────────────────────────────────────

    def _count_vehicles(self, frame: np.ndarray) -> int:
        """Run SSD and return total vehicle count in the frame."""
        if self._ssd_net is None:
            # Mock: random count scaled to look realistic
            return int(np.random.randint(1, 12))

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self._ssd_net.setInput(blob)
        detections = self._ssd_net.forward()

        threshold = self._cfg.get("confidence_threshold", 0.5)
        count = 0
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])
            if confidence >= threshold and class_id in _VEHICLE_CLASS_IDS:
                count += 1

        return count

    def _score_emergency_vehicle(self, frame: np.ndarray) -> float:
        """Return probability [0–1] that an emergency vehicle is in the frame."""
        if self._ev_model is None:
            return float(np.random.uniform(0.0, 0.15))

        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        prediction = self._ev_model.predict(tensor, verbose=0)
        # Assumes model output: [p_non_emergency, p_emergency]
        return float(prediction[0][1])
