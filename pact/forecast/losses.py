"""Asymmetric horizon-weighted Huber loss (Eq. 11–13)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pact.config import ForecastConfig


class AsymmetricHorizonHuber(nn.Module):
    """``L = (1/H) Σ_h w_h · c(e_h) · Huber_δ(e_h)``.

    ``e = ŷ − y``; ``c(e) = κ`` if ``e < 0`` else ``1`` (under-prediction
    costs more). ``w_h = softmax(−β(h − 1))``.
    """

    def __init__(self, *, kappa: float, beta: float, huber_delta: float) -> None:
        super().__init__()
        if kappa < 1.0:
            raise ValueError(f"kappa must be >= 1, got {kappa}")
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}")
        self.kappa = kappa
        self.beta = beta
        self.huber_delta = huber_delta

    @classmethod
    def from_config(cls, config: ForecastConfig) -> AsymmetricHorizonHuber:
        return cls(
            kappa=config.kappa, beta=config.beta, huber_delta=config.huber_delta
        )

    def horizon_weights(
        self, horizon: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        idx = torch.arange(horizon, device=device, dtype=dtype)
        return torch.softmax(-self.beta * idx, dim=0)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.dim() != 3:
            raise ValueError(f"expected (B, H, 2), got {tuple(pred.shape)}")
        error = pred - target
        huber = F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")
        cost = torch.where(
            error < 0.0, torch.full_like(error, self.kappa), torch.ones_like(error)
        )
        horizon = pred.size(1)
        weights = self.horizon_weights(horizon, pred.device, pred.dtype)
        weighted = cost * huber * weights.view(1, horizon, 1)
        per_h = weighted.mean(dim=(0, 2))
        return per_h.sum() / horizon
