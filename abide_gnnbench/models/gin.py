# abide_gnnbench/models/gin.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from torch_geometric.nn import GINConv, global_add_pool


class GIN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        eps: float = 0.0,
        train_eps: bool = True,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            mlp = nn.Sequential(
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(h, h),
            )
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(h))
            prev_dim = h
        self.classifier = nn.Linear(prev_dim, 1)
        self.dropout = dropout

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.classifier(x).view(-1)


def build_gin(input_dim: int, cfg: Dict[str, Any]) -> GIN:
    return GIN(
        input_dim=input_dim,
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        dropout=cfg.get("dropout", 0.3),
        eps=cfg.get("eps", 0.0),
        train_eps=cfg.get("train_eps", True),
    )
