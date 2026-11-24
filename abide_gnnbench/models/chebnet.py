# abide_gnnbench/models/chebnet.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from torch_geometric.nn import ChebConv, global_mean_pool


class ChebNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        K: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.convs.append(ChebConv(prev_dim, h, K=K))
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
        x = global_mean_pool(x, batch)
        return self.classifier(x).view(-1)


def build_chebnet(input_dim: int, cfg: Dict[str, Any]) -> ChebNet:
    return ChebNet(
        input_dim=input_dim,
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        K=cfg.get("K", 3),
        dropout=cfg.get("dropout", 0.3),
    )
