"""
src/models.py — Neural network architectures.

GestureLSTM:      Baseline camera-only LSTM classifier.
DeepFusionLSTM:   Multimodal deep feature fusion model (camera + IMU).

Deep fusion formula (Section 3 of thesis):
    h_f(k) = φ(W_vis · h_vis(k) + W_imu · h_imu(k) + b)
"""

import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    """
    Baseline LSTM for camera-only dynamic gesture classification.

    Input:  F_vis(t) — sequence of 63-dim MediaPipe landmark vectors.
    Output: Class logits.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 63) — batch of landmark sequences.
        Returns:
            logits: (B, num_classes)
        """
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


class DeepFusionLSTM(nn.Module):
    """
    Multimodal deep feature fusion model.

    Processes camera (F_vis) and IMU (F_imu) streams through separate
    LSTM branches, fuses them with learnable projection matrices, then
    passes the fused sequence through a third LSTM for final classification.

    Fusion gate per timestep k:
        h_f(k) = tanh(W_vis · h_vis(k) + W_imu · h_imu(k) + b)
    """

    def __init__(
        self,
        cam_dim: int = 63,
        imu_dim: int = 3,
        hidden_cam: int = 48,
        hidden_imu: int = 24,
        fusion_dim: int = 48,
        num_layers: int = 1,
        num_classes: int = 2,
    ):
        super().__init__()

        # Separate LSTMs for each modality
        self.lstm_vis = nn.LSTM(cam_dim, hidden_cam, num_layers, batch_first=True)
        self.lstm_imu = nn.LSTM(imu_dim, hidden_imu, num_layers, batch_first=True)

        # Learnable fusion projections: W_vis, W_imu
        self.W_vis = nn.Linear(hidden_cam, fusion_dim)
        self.W_imu = nn.Linear(hidden_imu, fusion_dim)
        self.phi = nn.Tanh()  # φ(·)

        # LSTM over fused sequence h_f(k)
        self.lstm_fusion = nn.LSTM(fusion_dim, fusion_dim, num_layers, batch_first=True)

        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(
        self, cam_seq: torch.Tensor, imu_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cam_seq: (B, T, 63) — F_vis(t), camera landmark sequences.
            imu_seq: (B, T, 3)  — F_imu(t), IMU context sequences.
        Returns:
            logits: (B, num_classes)
        """
        h_vis, _ = self.lstm_vis(cam_seq)   # (B, T, hidden_cam)
        h_imu, _ = self.lstm_imu(imu_seq)   # (B, T, hidden_imu)

        # Deep fusion: h_f(k) = φ(W_vis·h_vis(k) + W_imu·h_imu(k) + b)
        h_f = self.phi(self.W_vis(h_vis) + self.W_imu(h_imu))  # (B, T, fusion_dim)

        _, (h_last, _) = self.lstm_fusion(h_f)
        return self.classifier(h_last[-1])
