"""
AI-Enabled Traffic Management System
Entry point — wires together all modules and starts the main loop.

Usage:
    python src/main.py
    python src/main.py --config config/default.yaml
    python src/main.py --config config/default.yaml --demo   # run on sample video
"""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from utils.config_loader import load_config
from utils.logger import setup_logger
from detection.video_detector import VideoDetector
from detection.audio_detector import AudioDetector
from detection.ensemble import EnsembleEngine
from signaling.signal_controller import SignalController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Enabled Traffic Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on bundled sample video instead of live camera",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config & Logging ──────────────────────────────────────────────────────
    config = load_config(args.config)
    if args.log_level:
        config["logging"]["level"] = args.log_level

    logger = setup_logger(config["logging"])
    logger.info("Starting AI Traffic Management System")
    logger.info(f"Config loaded from: {args.config}")

    # ── Component Initialisation ──────────────────────────────────────────────
    video_detector = VideoDetector(config["detection"]["video"], demo=args.demo)
    audio_detector = AudioDetector(config["detection"]["audio"], config["audio"])
    ensemble = EnsembleEngine(config["ensemble"])
    controller = SignalController(config["signaling"], config["emergency"])

    # ── Graceful Shutdown ─────────────────────────────────────────────────────
    shutdown_event = threading.Event()

    def _handle_signal(signum, frame):
        logger.info("Shutdown signal received — stopping gracefully.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Main Loop ─────────────────────────────────────────────────────────────
    logger.info("System live. Press Ctrl+C to stop.")

    try:
        video_thread = threading.Thread(
            target=video_detector.run, args=(shutdown_event,), daemon=True
        )
        audio_thread = threading.Thread(
            target=audio_detector.run, args=(shutdown_event,), daemon=True
        )

        video_thread.start()
        audio_thread.start()

        while not shutdown_event.is_set():
            # Pull latest scores from each detector
            video_score, lane_counts = video_detector.get_latest()
            audio_score = audio_detector.get_latest()

            # Fuse and decide
            combined_score, is_emergency = ensemble.evaluate(video_score, audio_score)

            # Update signal timing
            controller.update(lane_counts, is_emergency)

            shutdown_event.wait(timeout=config["signaling"]["cycle_interval"])

    except Exception as exc:
        logger.exception(f"Fatal error in main loop: {exc}")
        sys.exit(1)
    finally:
        video_detector.release()
        audio_detector.release()
        logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
