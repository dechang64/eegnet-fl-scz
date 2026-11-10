"""PyTorch EEGNet (after Lawhern et al. 2018), kernel_length=64 targets N100.

AUDIT 2026-11-10
---------------
B1 (high): the previous `pooled_feat_t` closed form, ((T+1)//4)//8, accounted
    for block1's padded temporal conv (+1) but NOT for block2's temporal conv
    (kernel 16, padding 8), which adds another +1 before the second pooling.
    Verified: 72 values of T in [20, 600) disagree with the true feature length
    (T=27..30, 59..62, 91..94, ...), yielding either a bogus "too short"
    ValueError or a head/feature shape mismatch at forward time. Feature length
    is now *measured* with a dry-run forward pass, so it cannot drift again.
B8 (low): input tensor shape is now validated, and the depthwise kernel is
    constrained with max_norm=1 as in the reference implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pooled_feat_t(n_timesteps: int, pool1: int = 4, pool2: int = 8) -> int:
    """Closed form kept for tests/reference; EEGNet measures this directly."""
    after_pool1 = (n_timesteps + 1) // pool1      # block1 conv (pad k//2) -> +1
    after_conv2 = after_pool1 + 1                 # block2 conv(1,16,pad 8) -> +1
    return after_conv2 // pool2


class EEGNet(nn.Module):
    def __init__(self, n_classes=2, n_channels=5, n_timesteps=320,
                 dropout=0.25, kernel_length=64, F1=8, F2=16, D=2,
                 max_norm=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.max_norm = max_norm
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                   groups=F1, bias=False)
        self.block1_post = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        feat_t = self._measure_feat_t(n_channels, n_timesteps)
        if feat_t < 1:
            raise ValueError(
                f"n_timesteps={n_timesteps} too short for this architecture "
                f"(measured feature length {feat_t})")
        self.feat_t = feat_t
        self.head = nn.Linear(F2 * feat_t, n_classes)

    @torch.no_grad()
    def _measure_feat_t(self, n_channels: int, n_timesteps: int) -> int:
        was_training = self.training
        self.eval()                      # deterministic BN during the dry run
        try:
            z = self._body(torch.zeros(2, 1, n_channels, n_timesteps))
            return int(z.shape[-1])      # time axis only; head adds F2
        finally:
            self.train(was_training)

    def _body(self, x):
        return self.block2(self.block1_post(self.depthwise(self.block1(x))))

    def _apply_max_norm(self):
        if self.max_norm is not None:
            with torch.no_grad():
                self.depthwise.weight.copy_(torch.renorm(
                    self.depthwise.weight.detach(), p=2, dim=0,
                    maxnorm=self.max_norm))

    def forward(self, x):                      # x: (B, 1, C, T)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"expected input of shape (B, 1, C, T), got {tuple(x.shape)}")
        if x.shape[2] != self.n_channels:
            raise ValueError(
                f"channel mismatch: model built for {self.n_channels}, "
                f"input has {x.shape[2]}")
        self._apply_max_norm()
        return self.head(self._body(x).flatten(1))

    def forward_features(self, x):             # for XAI / probing
        self._apply_max_norm()
        return self._body(x).flatten(1)
