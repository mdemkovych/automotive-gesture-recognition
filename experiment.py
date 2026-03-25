"""
experiment.py — Full experimental pipeline.

Runs all experiments from Chapter 4 of the MSc thesis:
  1. Static gesture classification  (MLP on MediaPipe landmarks)
  2. Dynamic gesture classification (LSTM, camera-only, clean)
  3. Vibration robustness test      (LSTM, camera-only, noisy)
  4. Deep fusion evaluation         (DeepFusionLSTM, camera + IMU, noisy)
  5. Context FSM demo               (5 driving scenarios)
  6. Inference time benchmarks      (MLP and LSTM)

Usage:
    python experiment.py

Dataset must be placed in data/ before running.
See data/static/README.md and data/dynamic/README.md for structure.
"""

# ── Suppress MediaPipe / TensorFlow C++ logs ──────────────────
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_logtostderr"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)

_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

# ── Standard imports ───────────────────────────────────────────
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.stderr = _stderr  # Restore stderr after silent imports

# ── Project imports ────────────────────────────────────────────
from config.settings import BATCH_SIZE, EPOCHS, RANDOM_STATE, MAX_SEQ_LEN
from src.dataset import load_static, load_dynamic, DynamicGestureDataset, FusionGestureDataset
from src.models import GestureLSTM, DeepFusionLSTM
from src.imu_simulation import augment_with_vibration
from src.fsm import ContextFSM, DrivingState

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Static classifier ───────────────────────────────────────

def train_static(X, y, id_to_name) -> MLPClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu",
        solver="adam", max_iter=300, random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n[RESULT] Static MLP accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(
        y_test, y_pred,
        target_names=[id_to_name[i] for i in sorted(id_to_name)]
    ))
    print("[CONFUSION]\n", confusion_matrix(y_test, y_pred))
    return clf


# ── 2 & 3. Dynamic LSTM ────────────────────────────────────────

def _make_loader(X, y, shuffle: bool) -> DataLoader:
    return DataLoader(DynamicGestureDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_lstm(X_train, y_train, X_val, y_val, id_to_name) -> Tuple[GestureLSTM, float]:
    model = GestureLSTM(input_dim=X_train.shape[2],
                        num_classes=len(id_to_name)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n[INFO] Training GestureLSTM (camera-only, clean)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in _make_loader(X_train, y_train, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={total_loss / len(y_train):.4f}")

    acc = _eval_lstm(model, X_val, y_val, id_to_name, "LSTM (clean)")
    return model, acc


def _eval_lstm(model, X, y, id_to_name, title: str) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in _make_loader(X, y, shuffle=False):
            preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
            trues.extend(yb.numpy())
    acc = accuracy_score(trues, preds)
    print(f"\n[RESULT] {title}: {round(acc, 4)}")
    print(classification_report(trues, preds,
                                 target_names=[id_to_name[i] for i in sorted(id_to_name)]))
    print("[CONFUSION]\n", confusion_matrix(trues, preds))
    return acc


# ── 4. Deep Fusion LSTM ────────────────────────────────────────

def train_fusion(X_cam_tr, X_imu_tr, y_tr,
                 X_cam_val, X_imu_val, y_val,
                 id_to_name) -> Tuple[DeepFusionLSTM, float]:

    model = DeepFusionLSTM(
        cam_dim=X_cam_tr.shape[2], imu_dim=X_imu_tr.shape[2],
        num_classes=len(id_to_name),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    tr_ds  = FusionGestureDataset(X_cam_tr,  X_imu_tr,  y_tr)
    val_ds = FusionGestureDataset(X_cam_val, X_imu_val, y_val)

    print("\n[INFO] Training DeepFusionLSTM (camera + IMU, noisy)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xc, xi, yb in DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True):
            xc, xi, yb = xc.to(device), xi.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xc, xi), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xc.size(0)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={total_loss / len(y_tr):.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xc, xi, yb in DataLoader(val_ds, batch_size=BATCH_SIZE):
            preds.extend(torch.argmax(model(xc.to(device), xi.to(device)), 1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    print(f"\n[RESULT] DeepFusionLSTM (noisy+IMU): {round(acc, 4)}")
    print(classification_report(trues, preds,
                                 target_names=[id_to_name[i] for i in sorted(id_to_name)]))
    print("[CONFUSION]\n", confusion_matrix(trues, preds))
    return model, acc


# ── 5. FSM demo ────────────────────────────────────────────────

def run_fsm_demo():
    fsm = ContextFSM()
    scenarios = [
        # v,    along, alat,  vib,  gesture, conf
        (0.0,   0.1,   0.1,   0.2,  0,       0.70),  # STOP     → execute
        (15.0,  0.2,   0.1,   0.3,  2,       0.75),  # CRUISE   → execute
        (25.0,  0.3,   1.0,   0.5,  1,       0.90),  # MANEUVER → block
        (20.0,  0.2,   0.1,   1.5,  0,       0.78),  # HIGH_VIB → block (low conf)
        (20.0,  0.2,   0.1,   1.5,  2,       0.90),  # HIGH_VIB → block (not allowed)
    ]
    print("\n[FSM DEMO]")
    for i, (v, along, alat, vib, g, c) in enumerate(scenarios):
        fsm.update_state(v, along, alat, vib)
        y_exec, y_cmd = fsm.decide(g, c)
        print(f"  t={i} | state={fsm.state.name:<10} | gesture={g}, conf={c:.2f} "
              f"→ y_exec={y_exec}, y_cmd={y_cmd}")


# ── 6. Inference benchmarks ────────────────────────────────────

def benchmark_static(clf, sample):
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        clf.predict(sample.reshape(1, -1))
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\n[TIME] MLP  (static):  {sum(times)/len(times):.3f} ms avg over 50 runs")


def benchmark_lstm(model, sample):
    x = torch.from_numpy(sample[None]).float().to(device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    print(f"[TIME] LSTM (dynamic): {sum(times)/len(times):.3f} ms avg over 50 runs")


# ── Main pipeline ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  GESTURE RECOGNITION SYSTEM — EXPERIMENTS")
    print("=" * 55)

    # 1. Static
    X_st, y_st, st_names = load_static("data/static")
    clf = train_static(X_st, y_st, st_names)

    # 2. Dynamic — clean split
    X_dyn, y_dyn, dyn_names = load_dynamic("data/dynamic", MAX_SEQ_LEN)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dyn, y_dyn, test_size=0.2, random_state=RANDOM_STATE, stratify=y_dyn
    )
    lstm, acc_clean = train_lstm(X_tr, y_tr, X_val, y_val, dyn_names)

    # 3. Vibration noise — camera-only
    X_tr_noisy, IMU_tr   = augment_with_vibration(X_tr)
    X_val_noisy, IMU_val = augment_with_vibration(X_val)
    acc_noisy = _eval_lstm(lstm, X_val_noisy, y_val, dyn_names, "LSTM (noisy-camera)")

    # 4. Deep fusion — camera + IMU
    _, acc_fusion = train_fusion(
        X_tr_noisy, IMU_tr, y_tr,
        X_val_noisy, IMU_val, y_val,
        dyn_names,
    )

    # Summary
    print("\n[SUMMARY]")
    print(f"  Static MLP accuracy:             see report above")
    print(f"  LSTM clean accuracy:             {round(acc_clean, 4)}")
    print(f"  LSTM noisy (camera-only):        {round(acc_noisy, 4)}")
    print(f"  DeepFusionLSTM (camera+IMU):     {round(acc_fusion, 4)}")
    print(f"  Delta (noise drop):              {round(acc_clean - acc_noisy, 4)}")
    print(f"  Fusion gain:                     {round(acc_fusion - acc_noisy, 4)}")

    # 5. FSM demo
    run_fsm_demo()

    # 6. Benchmarks
    benchmark_static(clf, X_st[0])
    benchmark_lstm(lstm, X_val[0])

    print("\n" + "=" * 55)
    print("  Experiments complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
