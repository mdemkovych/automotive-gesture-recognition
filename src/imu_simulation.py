"""
src/imu_simulation.py — Synthetic IMU data generation and vibration augmentation.

Since physical IMU hardware was not available during development, these
functions simulate realistic automotive vibration profiles for training
and evaluation of the DeepFusionLSTM model.
"""

from typing import Tuple

import numpy as np

from config.settings import (
    VIB_AMPLITUDE,
    VIB_FREQUENCY,
    NOISE_ALPHA,
    NOISE_SIGMA,
    RANDOM_STATE,
)


def simulate_imu(T: int) -> np.ndarray:
    """
    Generate a synthetic F_imu(t) sequence representing automotive vibration context.

    The IMU signal captures environmental vibration (road surface, engine),
    independent of any specific hand gesture — it characterises the background,
    not the gesture itself.

    Args:
        T: Sequence length (number of timesteps).

    Returns:
        imu: float32 array of shape (T, 3) — simulated [ax, ay, az] readings.
    """
    t = np.linspace(0, 1, T)

    # Two-component sinusoidal vibration profile
    base = VIB_AMPLITUDE * (
        np.sin(2 * np.pi * VIB_FREQUENCY * t)
        + 0.7 * np.sin(2 * np.pi * (VIB_FREQUENCY * 0.5) * t + 0.7)
    )

    imu = np.stack([
        base + np.random.normal(0, 0.15, T),          # ax
        0.6 * base + np.random.normal(0, 0.15, T),    # ay
        1.1 * base + np.random.normal(0, 0.15, T),    # az
    ], axis=1)

    return imu.astype(np.float32)


def add_vibration_noise(seq: np.ndarray, imu_seq: np.ndarray) -> np.ndarray:
    """
    Corrupt a clean landmark sequence with vibration-proportional noise.

    Models the effect of vehicle vibration on camera-based hand tracking:
    stronger vibration → larger distortion of detected landmarks.

    Args:
        seq:     Clean landmark sequence, shape (T, 63).
        imu_seq: Corresponding IMU sequence, shape (T, 3).

    Returns:
        Noisy landmark sequence, shape (T, 63), float32.
    """
    T, D = seq.shape
    vib_strength = np.linalg.norm(imu_seq, axis=1, keepdims=True)   # (T, 1)
    vib_expanded = np.repeat(vib_strength, D, axis=1)               # (T, 63)

    noise = np.random.normal(0, NOISE_SIGMA, size=(T, D))

    # Additive noise scaled by vibration + small multiplicative drift
    seq_noisy = seq + NOISE_ALPHA * vib_expanded * noise
    seq_noisy = seq_noisy * (1.0 + 0.15 * np.tanh(vib_expanded))

    return seq_noisy.astype(np.float32)


def augment_with_vibration(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply vibration augmentation to an entire dataset split.

    For each sequence, generates a synthetic IMU profile and corrupts
    the camera landmarks proportionally.

    Args:
        X: Clean camera sequences, shape (N, T, 63).

    Returns:
        X_noisy: Vibration-corrupted sequences, shape (N, T, 63).
        X_imu:   Corresponding IMU sequences,   shape (N, T, 3).
    """
    noisy_list, imu_list = [], []
    for seq in X:
        imu = simulate_imu(seq.shape[0])
        noisy_list.append(add_vibration_noise(seq, imu))
        imu_list.append(imu)

    return np.stack(noisy_list), np.stack(imu_list)
