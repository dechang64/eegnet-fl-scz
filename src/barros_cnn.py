"""1D-CNN baseline in the spirit of Barros 2022 (APPROXIMATION, not a
faithful reimplementation — transcribe exact config from the paper at W1).
Input: (B, C, T); train.py converts.
"""
from __future__ import annotations
import torch.nn as nn
 
 
class BarrosCNN(nn.Module):
    def __init__(self, n_classes=2, n_channels=5, n_timesteps=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, 16, padding=8),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 4, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, n_classes)
 
    def forward(self, x):                      # x: (B, C, T)
        return self.head(self.net(x).squeeze(-1))
