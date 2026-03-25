"""
src/skeleton.py — Hand skeleton extraction using MediaPipe.

Extracts 21 3D hand landmarks (x, y, z) from images and video frames,
producing a 63-dimensional feature vector F_vis(t) per frame.
"""

from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands


def extract_from_image(image_path: str) -> Optional[np.ndarray]:
    """
    Extract 63-dim landmark vector from a static image file.

    Args:
        image_path: Path to .jpg / .png image.

    Returns:
        Float32 array of shape (63,) — [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
        or None if no hand is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        result = hands.process(img_rgb)
        if not result.multi_hand_landmarks:
            return None

        coords = []
        for lm in result.multi_hand_landmarks[0].landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)


def extract_from_frame(frame: np.ndarray, hands) -> Optional[np.ndarray]:
    """
    Extract 63-dim landmark vector from a single video frame.

    Args:
        frame:  BGR frame (numpy array) from cv2.VideoCapture.
        hands:  An active mediapipe Hands context (reused across frames
                for efficiency and temporal tracking).

    Returns:
        Float32 array of shape (63,), or None if no hand detected.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    coords = []
    for lm in result.multi_hand_landmarks[0].landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)
