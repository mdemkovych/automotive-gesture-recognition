"""
src/dataset.py — Dataset loading and PyTorch Dataset wrappers.

Loads static images and dynamic video sequences, extracts MediaPipe
landmarks, and wraps them in PyTorch-compatible Dataset classes.
"""

import glob
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp
import torch
from torch.utils.data import Dataset

from src.skeleton import extract_from_image, extract_from_frame
from config.settings import MAX_SEQ_LEN

mp_hands = mp.solutions.hands


# ── Loaders ────────────────────────────────────────────────────

def load_static(static_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load static gesture images and extract 63-dim landmark vectors.

    Expects directory structure:
        static_dir/
            fist/       ← *.jpg
            palm/       ← *.jpg
            pointing/   ← *.jpg

    Returns:
        X: float32 array of shape (N, 63)
        y: int64 label array of shape (N,)
        id_to_name: dict mapping class index → class name
    """
    class_names = sorted(
        d for d in os.listdir(static_dir)
        if os.path.isdir(os.path.join(static_dir, d))
    )
    if not class_names:
        raise RuntimeError(f"No class subdirectories found in {static_dir}")

    id_to_name = {i: name for i, name in enumerate(class_names)}
    name_to_id = {name: i for i, name in id_to_name.items()}

    X_list, y_list = [], []
    for class_name in class_names:
        paths = glob.glob(os.path.join(static_dir, class_name, "*.*"))
        print(f"[INFO] Static class '{class_name}': {len(paths)} images")
        for path in paths:
            vec = extract_from_image(path)
            if vec is not None:
                X_list.append(vec)
                y_list.append(name_to_id[class_name])

    if not X_list:
        raise RuntimeError("Static dataset is empty. Check data/static/*/")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    print(f"[INFO] Static dataset loaded: {X.shape[0]} samples, {len(class_names)} classes")
    return X, y, id_to_name


def load_dynamic(
    dynamic_dir: str,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load dynamic gesture videos and extract per-frame landmark sequences.

    Expects directory structure:
        dynamic_dir/
            left/   ← *.mov
            right/  ← *.mov

    Each video is converted to a fixed-length sequence of shape (max_seq_len, 63)
    using padding (edge mode) or truncation.

    Returns:
        X: float32 array of shape (N, T, 63)
        y: int64 label array of shape (N,)
        id_to_name: dict mapping class index → class name
    """
    class_names = sorted(
        d for d in os.listdir(dynamic_dir)
        if os.path.isdir(os.path.join(dynamic_dir, d))
    )
    if not class_names:
        raise RuntimeError(f"No class subdirectories found in {dynamic_dir}")

    id_to_name = {i: name for i, name in enumerate(class_names)}
    name_to_id = {name: i for i, name in id_to_name.items()}

    X_list, y_list = [], []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for class_name in class_names:
            paths = glob.glob(os.path.join(dynamic_dir, class_name, "*.*"))
            print(f"[INFO] Dynamic class '{class_name}': {len(paths)} videos")

            for path in paths:
                cap = cv2.VideoCapture(path)
                seq = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    vec = extract_from_frame(frame, hands)
                    if vec is not None:
                        seq.append(vec)
                cap.release()

                if not seq:
                    print(f"[WARN] No landmarks found: {path}")
                    continue

                seq = np.stack(seq)  # (T, 63)

                # Pad or truncate to fixed length
                T = seq.shape[0]
                if T > max_seq_len:
                    seq = seq[:max_seq_len]
                elif T < max_seq_len:
                    seq = np.pad(seq, ((0, max_seq_len - T), (0, 0)), mode="edge")

                X_list.append(seq)
                y_list.append(name_to_id[class_name])

    if not X_list:
        raise RuntimeError("Dynamic dataset is empty. Check data/dynamic/*/")

    X = np.stack(X_list)  # (N, T, 63)
    y = np.array(y_list, dtype=np.int64)
    print(f"[INFO] Dynamic dataset loaded: {X.shape[0]} sequences, {len(class_names)} classes")
    return X, y, id_to_name


# ── PyTorch Dataset wrappers ───────────────────────────────────

class DynamicGestureDataset(Dataset):
    """Camera-only dynamic gesture dataset: returns (sequence, label)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class FusionGestureDataset(Dataset):
    """Multimodal dataset for DeepFusionLSTM: returns (cam_seq, imu_seq, label)."""

    def __init__(self, X_cam: np.ndarray, X_imu: np.ndarray, y: np.ndarray):
        assert X_cam.shape[0] == X_imu.shape[0] == y.shape[0]
        self.X_cam = torch.from_numpy(X_cam).float()
        self.X_imu = torch.from_numpy(X_imu).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_cam[idx], self.X_imu[idx], self.y[idx]
