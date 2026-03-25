# Adaptive Gesture Recognition for Automotive IoT

> A multimodal deep learning system for robust, contactless gesture control in automotive environments — combining camera-based hand tracking with inertial sensing to maintain accuracy under vibration.

## Overview

This project implements an adaptive gesture recognition pipeline designed for in-vehicle use. The system fuses visual (camera) and inertial (IMU) sensor streams using a deep feature fusion model, and applies a Finite State Machine (FSM) layer that adapts command behaviour based on driving context (speed, acceleration, vibration level).

| Metric | Result |
|---|---|
| Static gesture accuracy (MLP) | **98.78%** |
| Dynamic accuracy — camera only | ~88% under vibration |
| Dynamic accuracy — camera + IMU fusion | **~94%** under vibration |
| LSTM inference time (32 frames) | ~0.47 ms |

---

## Key Features

- **Multimodal sensor fusion** — camera + IMU compensate each other's weaknesses (lighting vs vibration drift)
- **DeepFeatureFusion** — learnable fusion weights (W_vis, W_imu) via dual-branch LSTM
- **Context-aware FSM** — blocks or requires confirmation of gestures based on vehicle state
- **Embedded deployment target** — optimised for Raspberry Pi 5 + Hailo-8L NPU (13 TOPS)
- **Real-time capable** — classifier latency well under 150ms system requirement

---

## System Architecture

![System architecture](docs/images/system-architecture.png)

The hardware stack consists of two sensor inputs (Raspberry Pi Camera Module 3 NoIR + IMU MPU-6050), a Raspberry Pi 5 with Hailo-8L NPU (13 TOPS) for inference, and two output interfaces — Bluetooth 5.0 BLE for basic mode and CAN Bus (MCP2515) for full automotive integration.

### Schematic system representation 

```
┌─────────────────────────────────────────────────────┐
│                   Sensor Layer                      │
│  Raspberry Pi Camera Module 3 (120°, NoIR, CSI-2)   │
│  IMU MPU-6050 (3-axis accel + 3-axis gyro, I²C)     │
└────────────────┬──────────────────┬─────────────────┘
                 │                  │
         Visual stream         IMU stream
                 │                  │
┌────────────────▼──────────────────▼─────────────────┐
│              Processing Layer                       │
│  MediaPipe → 21 keypoints → 63-dim vector           │
│  Complementary filter: θ = α(θ_prev + ω·Δt)         │
│              + (1-α)·θ_acc                          │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│           DeepFeatureFusion (PyTorch)               │
│  LSTM_cam (63→64) + LSTM_imu (3→32)                 │
│  h_f = tanh(W_vis·h_cam + W_imu·h_imu + b)          │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│            Context FSM                              │
│  States: STOP / CRUISE / MANEUVER / HIGH_VIB        │
│  Filters or confirms gestures per driving state     │
└──────────────────────────┬──────────────────────────┘
                           │
                    Command output
```

---

## Pipeline

![Gesture recognition pipeline](docs/images/pipeline.png)

---

## Data Processing Flow

![Multimodal data processing flow](docs/images/data-processing-flow.png)

The system processes two parallel sensor streams. Visual data goes through Gaussian preprocessing and MediaPipe skeletonisation to produce F_vis(t). Inertial data is stabilised with a complementary filter to produce F_imu(t). Both streams are fused via the deep fusion gate before classification.

---

## Dataset

### Static gestures — 3 classes, ~272 samples

![Static dataset](docs/images/static-dataset.png)

### Dynamic gestures — 2 classes, ~118 sequences

![Dynamic dataset](docs/images/dynamic-dataset.png)

All sequences are padded or truncated to 32 frames.  
**Note:** Dataset not included in this repository (personal data). See [`data/static/README.md`](data/static/README.md) and [`data/dynamic/README.md`](data/dynamic/README.md) for expected folder structure.

---

## Models

### Static classifier (MLP)
- Input: 63-dim MediaPipe landmark vector (21 keypoints × 3 coords)
- Classes: palm / fist / pointing
- Accuracy: **98.78%**, F1 > 0.97 for all classes
- Inference: ~0.065 ms per vector

### Dynamic classifier — baseline (GestureLSTM)
- Input: sequence of 63-dim vectors, padded to 32 frames
- Classes: swipe left / swipe right
- Accuracy: ~96% clean, drops to ~88% under vibration noise

### Dynamic classifier — fusion (DeepFusionLSTM)
- Dual-branch LSTM: visual branch (63 → 48) + IMU branch (3 → 24)
- Deep fusion gate per timestep k:
  ```
  h_f(k) = tanh(W_vis · h_vis(k) + W_imu · h_imu(k) + b)
  ```
- Accuracy: **~94%** under vibration (+6pp vs camera-only)

---

## Context FSM

The FSM layer sits above the classifier and filters gesture commands based on driving state:

| State | Allowed gestures | Min confidence |
|---|---|---|
| STOP | all (0–3) | 0.60 |
| CRUISE | 0, 1, 2 | 0.70 |
| MANEUVER | none | — |
| HIGH_VIB | 0, 1 | 0.80 |

---

## Project Structure

```
automotive-gesture-recognition/
│
├── experiment.py              # Entry point — run this
├── requirements.txt
│
├── config/
│   └── settings.py            # All hyperparameters and paths
│
├── src/
│   ├── skeleton.py            # MediaPipe landmark extraction
│   ├── dataset.py             # Data loaders + PyTorch Dataset classes
│   ├── models.py              # GestureLSTM + DeepFusionLSTM
│   ├── imu_simulation.py      # Synthetic IMU + vibration augmentation
│   └── fsm.py                 # Context-aware Finite State Machine
│
├── data/
│   ├── static/                # Static gesture images (not committed)
│   └── dynamic/               # Dynamic gesture videos (not committed)
│
└── docs/
    └── images/                # Architecture diagrams and dataset figures
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your dataset

Follow the instructions in [`data/static/README.md`](data/static/README.md) and [`data/dynamic/README.md`](data/dynamic/README.md).

### 3. Run experiments

```bash
python experiment.py
```

Runs the full pipeline: static classification → dynamic LSTM → vibration robustness → deep fusion → FSM demo → inference benchmarks.

---

## Results Summary

| Condition | Accuracy |
|---|---|
| Static gestures (MLP) | 98.78% |
| Dynamic, clean, camera-only | ~96% |
| Dynamic, vibration noise, camera-only | ~88% |
| Dynamic, vibration noise, camera + IMU | **~94%** |

---

## Limitations

- CPU-only testing (target hardware Raspberry Pi 5 + Hailo-8L NPU not yet benchmarked)
- IMU data is synthetic — generated noise profiles, not from a physical sensor
- Small dataset collected in controlled conditions
- No real in-vehicle testing performed

---

## Skills Demonstrated

`PyTorch` · `LSTM` · `CNN` · `MediaPipe` · `Sensor fusion` · `Signal processing` · `Finite State Machine` · `Embedded ML` · `Python` · `Real-time systems` · `Automotive IoT` · `NumPy` · `scikit-learn`

---

## Project Context

**Degree:** MSc in Computer Science — Computer Control Systems for Mobile Objects (Automotive)  
**University:** Lviv Polytechnic National University  
**Year:** 2025  
**Thesis title:** *"Adaptive Dynamic Gesture Recognition System for Contactless Control in Automotive IoT Environments"*
