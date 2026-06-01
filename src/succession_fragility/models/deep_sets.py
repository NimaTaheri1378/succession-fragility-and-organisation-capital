from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSetConfig:
    n_features: int
    hidden_dim: int = 128
    dropout: float = 0.10


def build_deep_set_model(config: DeepSetConfig):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Deep Sets model.") from exc

    class DeepSetRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(config.n_features, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.rho = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 2),
            )

        def forward(self, x, mask=None):
            encoded = self.phi(x)
            if mask is not None:
                encoded = encoded * mask.unsqueeze(-1)
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = encoded.sum(dim=1) / denom
            else:
                pooled = encoded.mean(dim=1)
            return self.rho(pooled)

    return DeepSetRegressor()
