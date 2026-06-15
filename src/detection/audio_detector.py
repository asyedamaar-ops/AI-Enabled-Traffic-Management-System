"""
Audio Detector
==============
Continuously captures microphone audio, converts 3-second windows into
MFCC + spectrogram features, and feeds them into a trained CNN to produce
a siren probability score [0.0 - 1.0].

Runs in its own background thread. Main loop polls get_latest().
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("traffic_mgmt.audio_detector")


class AudioDetector:
    """CNN-based emergency vehicle siren detector from live microphone audio."""

    def __init__(self, audio_cfg: dict, capture_cfg: dict) -> None:
        self._audio_cfg = audio_cfg
        self._capture_cfg = capture_cfg

        self._lock = threading.Lock()
        self._latest_score: float = 0.0

        self._model = None
        self._stream = None
        self._pyaudio = None

        self._sample_buffer: np.ndarray = np.array([], dtype=np.float32)

        self._load_model()
        self._open_stream()

    # -- Initialisation -------------------------------------------------------

    def _load_model(self) -> None:
        model_path = Path(self._audio_cfg.get("model_path", "models/cnn_siren.h5"))
        if model_path.exists():
            try:
                from tensorflow import keras
                self._model = keras.models.load_model(str(model_path))
                logger.info("CNN siren detection model loaded.")
            except Exception as exc:
                logger.warning(f"Could not load audio model ({exc}). Using mock scores.")
        else:
            logger.warning(f"Audio model not found at {model_path}. Using mock scores.")

    def _open_stream(self) -> None:
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            device_index = self._capture_cfg.get("device_index", None)
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self._capture_cfg.get("channels", 1),
                rate=self._capture_cfg.get("sample_rate", 44100),
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self._capture_cfg.get("chunk_size", 1024),
            )
            logger.info("Microphone stream opened.")
        except Exception as exc:
            logger.warning(f"Could not open audio stream ({exc}). Running in mock mode.")
            self._stream = None

    # -- Public API -----------------------------------------------------------

    def run(self, shutdown_event: threading.Event) -> None:
        """Audio capture + inference loop — runs in a background thread."""
        sample_rate = self._capture_cfg.get("sample_rate", 44100)
        clip_duration = self._audio_cfg.get("clip_duration", 3)
        overlap = self._audio_cfg.get("overlap", 1)
        chunk_size = self._capture_cfg.get("chunk_size", 1024)

        clip_samples = sample_rate * clip_duration
        step_samples = sample_rate * (clip_duration - overlap)

        while not shutdown_event.is_set():
            if self._stream is None:
                score = float(np.random.uniform(0.0, 0.12))
                with self._lock:
                    self._latest_score = score
                shutdown_event.wait(timeout=clip_duration - overlap)
                continue

            try:
                raw = self._stream.read(chunk_size, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.float32)
                self._sample_buffer = np.concatenate([self._sample_buffer, samples])
            except Exception as exc:
                logger.debug(f"Audio read error: {exc}")
                time.sleep(0.05)
                continue

            if len(self._sample_buffer) >= clip_samples:
                clip = self._sample_buffer[:clip_samples]
                self._sample_buffer = self._sample_buffer[step_samples:]
                score = self._classify_clip(clip, sample_rate)
                with self._lock:
                    self._latest_score = score

    def get_latest(self) -> float:
        """Return the most recent siren probability score."""
        with self._lock:
            return self._latest_score

    def release(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio is not None:
            self._pyaudio.terminate()
        logger.info("Audio detector released.")

    # -- Feature Extraction + Inference ---------------------------------------

    def _classify_clip(self, clip: np.ndarray, sample_rate: int) -> float:
        try:
            features = self._extract_features(clip, sample_rate)
        except Exception as exc:
            logger.debug(f"Feature extraction failed: {exc}")
            return 0.0

        if self._model is None:
            return float(np.random.uniform(0.0, 0.15))

        try:
            tensor = features[np.newaxis, ..., np.newaxis].astype(np.float32)
            prediction = self._model.predict(tensor, verbose=0)
            return float(prediction[0][1])
        except Exception as exc:
            logger.debug(f"Model inference failed: {exc}")
            return 0.0

    def _extract_features(self, clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a 2-D MFCC array (n_mfcc x time_frames)."""
        import librosa

        n_mfcc = self._audio_cfg.get("n_mfcc", 40)
        n_fft = self._audio_cfg.get("n_fft", 2048)
        hop_length = self._audio_cfg.get("hop_length", 512)

        mfcc = librosa.feature.mfcc(
            y=clip,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        return mfcc
