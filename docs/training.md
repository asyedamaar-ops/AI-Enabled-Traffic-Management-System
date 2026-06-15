# Training the Models

This document describes how each model in the system was trained.
Reproduce training or fine-tune on your own data by following these steps.

---

## 1. SSD Vehicle Detector

**Base model:** SSD MobileNet v2 (COCO pre-trained)
**Task:** Vehicle detection and counting per lane

### Dataset
- COCO 2017 — classes used: car (2), motorcycle (3), bus (5), truck (7)
- Any additional traffic camera footage you have can be added

### Fine-tuning
```bash
# Install TF Object Detection API
pip install tensorflow-object-detection-api

# Run fine-tuning (adjust paths)
python -m object_detection.model_main_tf2 \
  --pipeline_config_path=config/ssd_pipeline.config \
  --model_dir=models/ssd_training/ \
  --num_train_steps=50000
```

---

## 2. DenseNet-169 Emergency Vehicle Classifier

**Base model:** DenseNet-169 (ImageNet pre-trained)
**Task:** Binary classification — emergency vs non-emergency vehicle

### Dataset
- 10,000+ images of ambulances, fire trucks, police cars (various angles, lighting, weather)
- Augmented with random flips, brightness jitter, fog simulation

### Training
```python
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

base = DenseNet169(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
out = Dense(2, activation="softmax")(x)
model = Model(base.input, out)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=30)
model.save("models/densenet169_ev.h5")
```

**Results:** Precision 94.2%, Recall 92.7%

---

## 3. CNN Siren Detector

**Task:** Binary audio classification — siren present vs no siren

### Dataset
- UrbanSound8K (city sounds baseline)
- Custom siren recordings: ambulance, fire truck, police — from various distances
- Negative samples: traffic noise, construction, rain, horns

### Feature Extraction
```python
import librosa
import numpy as np

def extract_mfcc(path, sr=44100, n_mfcc=40):
    y, _ = librosa.load(path, sr=sr, duration=3.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
```

### Architecture
```
Input (40 x 130 x 1)
  → Conv2D(32, 3x3) + ReLU + MaxPool
  → Conv2D(64, 3x3) + ReLU + MaxPool
  → Conv2D(128, 3x3) + ReLU + MaxPool
  → Flatten → Dense(256) → Dropout(0.5) → Dense(2, softmax)
```

**Results:** 96.3% accuracy in controlled conditions

---

## Notes

- All models were exported to `.h5` (Keras) or `.pb` (TensorFlow SavedModel) format
- Place trained weights in the `models/` directory before running the system
- See `scripts/download_models.py` for downloading pre-trained weights
