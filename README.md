# 🚦 AI-Enabled Traffic Management System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.7-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Real-time, AI-powered traffic signal control with automatic emergency vehicle prioritization.**

*Final Year Project — B.Tech CSE (AI & ML), SRM IST Chennai | AIoT Course (21CSE292P)*

[Features](#-features) • [Architecture](#-architecture) • [Quickstart](#-quickstart) • [Results](#-results) • [Tech Stack](#-tech-stack)

</div>

---

## The Problem

Urban intersections still run on schedules written years ago. Fixed green/red cycles don't know whether the lane is empty or jammed. And when an ambulance is stuck behind a red light, seconds matter.

This project replaces static signal timing with a live, vision-and-audio-driven system that reads the road and responds to it — including stepping aside automatically for emergency vehicles.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🎥 **Real-time Vehicle Detection** | SSD model (91.3% mAP) counts vehicles per lane at 19 FPS |
| 🚑 **Emergency Vehicle Detection** | Dual-mode: visual (DenseNet-169, 94.2% precision) + audio siren (CNN, 96.3% accuracy) |
| 🔀 **Ensemble Decision Fusion** | Weighted combination of video + audio scores for 97.2% combined accuracy |
| ⚡ **Dynamic Signal Control** | Green time auto-adjusts based on live density — no manual tuning |
| 🖥️ **Live Dashboard** | Streamlit web dashboard with lane stats, alerts, and override controls |
| 🐳 **Docker Ready** | Single command to spin up the full system |

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│    📷 Traffic Cameras         🎙️ Roadside Microphones   │
└────────────────┬──────────────────────┬─────────────────┘
                 │                      │
         ┌───────▼──────┐      ┌────────▼───────┐
         │  Video       │      │  Audio         │
         │  Pipeline    │      │  Pipeline      │
         │  (OpenCV)    │      │  (Librosa)     │
         └───────┬──────┘      └────────┬───────┘
                 │                      │
    ┌────────────▼──┐         ┌─────────▼──────────┐
    │   SSD Model   │         │  CNN (MFCC +       │
    │ Vehicle Count │         │  Spectrogram)      │
    │  per Lane     │         │  Siren Detection   │
    └───────┬───────┘         └─────────┬──────────┘
            │                           │
            │    ┌──────────────────┐   │
            └───►│  Ensemble Engine │◄──┘
                 │  (Weighted Avg)  │
                 └────────┬─────────┘
                          │
               ┌──────────▼──────────┐
               │   Decision Engine   │
               │  Normal  │Emergency │
               │  Timing  │Override  │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  Signal Controller  │
               │  + Dashboard        │
               └─────────────────────┘
```

---

## 🚀 Quickstart

### Option A — Docker (Recommended)

```bash
git clone https://github.com/yourusername/aiot-traffic-management.git
cd aiot-traffic-management
docker-compose up
```

The dashboard opens at `http://localhost:8501`.

### Option B — Local Setup

```bash
# 1. Clone and enter
git clone https://github.com/yourusername/aiot-traffic-management.git
cd aiot-traffic-management

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the system
python src/main.py --config config/default.yaml

# 5. Open the dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
aiot-traffic-management/
│
├── src/
│   ├── main.py                    # Entry point
│   ├── detection/
│   │   ├── video_detector.py      # SSD + DenseNet-169 vehicle/EV detection
│   │   ├── audio_detector.py      # CNN siren detection via MFCC + spectrograms
│   │   └── ensemble.py            # Weighted fusion of video + audio scores
│   ├── signaling/
│   │   ├── signal_controller.py   # Core green/red timing logic
│   │   └── optimizer.py           # Density → optimal timing algorithm
│   ├── dashboard/
│   │   └── app.py                 # Streamlit real-time dashboard
│   └── utils/
│       ├── logger.py
│       └── config_loader.py
│
├── config/
│   └── default.yaml               # All tunable parameters in one file
│
├── models/                        # Pre-trained weights (see Models section)
│   ├── ssd_vehicle.pb
│   ├── densenet169_ev.h5
│   └── cnn_siren.h5
│
├── tests/
│   ├── test_video_detector.py
│   ├── test_audio_detector.py
│   └── test_ensemble.py
│
├── scripts/
│   └── download_models.py         # Downloads model weights automatically
│
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 🧠 How It Works

### Dynamic Traffic Signaling

Every second, cameras at each lane send a frame to the SSD detection model. The model counts vehicles. The optimizer maps those counts to a proportional green-light duration — busier lane, longer green. Simple constraint: no lane waits more than 90 seconds, no lane gets less than 10.

```python
# Simplified core logic
green_time = base_time + (vehicle_count / max_count) * extra_time
green_time = max(MIN_GREEN, min(MAX_GREEN, green_time))
```

### Emergency Vehicle Detection

Two models run in parallel on every cycle:

- **Video** — DenseNet-169 looks for ambulance/fire truck body shape + flashing light patterns. Trained on 10,000+ images across lighting and weather conditions.
- **Audio** — CNN processes 3-second overlapping audio windows converted to spectrograms. Trained to distinguish sirens from horns, engines, and general city noise.

Their probability scores are fused:

```python
combined = (w_video * p_video) + (w_audio * p_audio)
if combined > THRESHOLD:
    trigger_emergency_override()
```

Weights shift dynamically — night or heavy rain nudges weight toward audio; clear daylight shifts it toward video.

### Green Corridor

When an emergency vehicle is confirmed, the system pre-clears its path: the intersection ahead goes green, adjacent lanes hold red, and the dashboard fires a visual alert. Average detection-to-signal-change: **3.2 seconds**.

---

## 📊 Results

Tested over four weeks at a simulated busy four-way intersection.

### Traffic Flow

| Metric | Improvement |
|--------|------------|
| Average waiting time | ↓ 31.7% during peak hours |
| Vehicle throughput | ↑ 24.3% |
| Average queue length | ↓ 27.5% |

### Emergency Vehicle Detection

| Metric | Value |
|--------|-------|
| Combined accuracy (video + audio) | **97.2%** |
| False positive rate | 1.3% |
| False negative rate | 1.5% |
| Detection → signal change | **3.2 seconds** |
| Emergency vehicle travel time (peak) | ↓ **42.1%** |

### System Reliability

- **99.7% uptime** over the full 4-week test period
- Processing latency: < 100ms per cycle on standard hardware

---

## ⚙️ Configuration

All parameters live in `config/default.yaml` — no hardcoded magic numbers in the code.

```yaml
signaling:
  min_green_seconds: 10
  max_green_seconds: 90
  base_green_seconds: 30
  cycle_interval: 1.0        # seconds between density recalculations

detection:
  video:
    fps: 10
    confidence_threshold: 0.5
    model_path: "models/densenet169_ev.h5"
  audio:
    sample_rate: 44100
    clip_duration: 3          # seconds
    overlap: 1                # seconds
    confidence_threshold: 0.6
    model_path: "models/cnn_siren.h5"

ensemble:
  emergency_threshold: 0.85
  video_weight_day: 0.6
  audio_weight_day: 0.4
  video_weight_night: 0.35
  audio_weight_night: 0.65

dashboard:
  port: 8501
  refresh_interval: 0.5      # seconds
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8 |
| Computer Vision | OpenCV 4.5.4, TensorFlow 2.7 |
| Detection Models | SSD (COCO), DenseNet-169 |
| Audio Processing | Librosa 0.9.1, PyAudio 0.2.11 |
| Deep Learning | PyTorch 1.10, Keras 2.7 |
| Dashboard | Streamlit 1.5 |
| Control UI | PyQt5 5.15.6 |
| Containerization | Docker 20.10.12 |

---

## 📥 Models

Model weights are not committed to the repo (too large). Download them with:

```bash
python scripts/download_models.py
```

Or manually place pre-trained weights in the `models/` directory. See `scripts/download_models.py` for links and expected filenames.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap

- [ ] YOLOv8 swap-in for faster inference
- [ ] Multi-intersection coordination (green wave)
- [ ] Edge deployment (Jetson Nano / Raspberry Pi)
- [ ] Weather-aware automatic weight tuning
- [ ] SUMO traffic simulator integration for large-scale testing

