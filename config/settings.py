# config/settings.py — Central configuration for all experiments.
# Edit paths and hyperparameters here before running.

import os

# ── Data paths ─────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
STATIC_DIR    = os.path.join(DATA_DIR, "static")
DYNAMIC_DIR   = os.path.join(DATA_DIR, "dynamic")

# ── Sequence settings ──────────────────────────────────────────
MAX_SEQ_LEN = 32     # All dynamic sequences padded/truncated to this length

# ── Training hyperparameters ───────────────────────────────────
BATCH_SIZE   = 4
EPOCHS       = 8
RANDOM_STATE = 42

# ── Vibration simulation ───────────────────────────────────────
VIB_AMPLITUDE = 1.8   # Base vibration amplitude for IMU simulation
VIB_FREQUENCY = 10.0  # Base vibration frequency (Hz)
NOISE_ALPHA   = 1.6   # Vibration influence on skeleton landmarks
NOISE_SIGMA   = 0.20  # Random noise standard deviation

# ── FSM thresholds ─────────────────────────────────────────────
FSM_V_STOP   = 0.5    # Speed threshold for STOP state (m/s)
FSM_A_STAB   = 0.3    # Acceleration threshold for stable cruise (m/s²)
FSM_A_MAN    = 0.8    # Lateral acceleration threshold for MANEUVER (m/s²)
FSM_VIB_HIGH = 1.2    # Vibration threshold for HIGH_VIB state
