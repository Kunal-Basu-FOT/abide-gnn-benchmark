# abide_gnnbench/models/mlp.py

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, List


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def build_mlp(input_dim: int, cfg: Dict[str, Any]) -> MLP:
    return MLP(
        input_dim=input_dim,
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        dropout=cfg.get("dropout", 0.3),
        use_batchnorm=cfg.get("use_batchnorm", True),
    )
