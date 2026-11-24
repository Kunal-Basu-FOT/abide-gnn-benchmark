# abide_gnnbench/models/gat.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn.norm import GraphNorm


def _default_hidden_dims(num_layers: int) -> List[int]:
    """
    Derive a hidden-dimension schedule for an arbitrary number of GAT layers,
    preserving the baseline's width strategy:

        - All intermediate layers : 128 units (multi-head, concatenated)
        - Final layer             : 64 units (single-head, feeds classifier)

    This matches the current 2-layer baseline hidden_dims=[128, 64] exactly
    when num_layers == 2, so existing experiments are unaffected.

    Args:
        num_layers: Number of GATConv layers to construct (>= 1).

    Returns:
        List of output widths, one per layer, length == num_layers.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_layers == 1:
        return [64]
    return [128] * (num_layers - 1) + [64]


def _build_norm_layer(normalization: str, dim: int) -> nn.Module:
    """
    Factory for the per-layer normalization module used after each GATConv.

    Args:
        normalization: One of "none", "batch", "layer", "graph".
        dim: Feature dimension the normalization layer operates over
            (== that GATConv layer's output width).

    Returns:
        An nn.Module implementing the requested normalization. All four
        variants accept a [N, dim] node-feature tensor as their primary
        input; GraphNorm additionally requires the PyG `batch` vector at
        call time (handled in GAT.forward, not here).

    Raises:
        ValueError: If `normalization` is not one of the supported options.
    """
    normalization = normalization.lower()
    if normalization == "none":
        return nn.Identity()
    elif normalization == "batch":
        return nn.BatchNorm1d(dim)
    elif normalization == "layer":
        return nn.LayerNorm(dim)
    elif normalization == "graph":
        return GraphNorm(dim)
    else:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            f"Expected one of: 'none', 'batch', 'layer', 'graph'."
        )


class GAT(nn.Module):
    """
    Graph Attention Network for brain-graph (ROI-level) classification.

    Architecture (for L layers):

        Input [N, F_in]
          -> GATConv_1 (heads=H, concat=True)  -> [N, H*128]
          -> (optional residual add, if shapes match)
          -> Norm (none / batch / layer / graph) -> LeakyReLU(0.2) -> Dropout
          ...
          -> GATConv_{L-1} (heads=H, concat=True) -> [N, H*128]
          -> (optional residual add, if shapes match)
          -> Norm -> LeakyReLU(0.2) -> Dropout
          -> GATConv_L (heads=1, concat=False)  -> [N, 64]
          -> (optional residual add, if shapes match)
          -> Norm -> LeakyReLU(0.2) -> Dropout
          -> global_mean_pool -> [B, 64]
          -> Linear -> [B] (raw logit)

    NOTE ON ACTIVATION: the current repository baseline uses LeakyReLU(0.2),
    NOT ELU, despite ELU being referenced in the project documentation
    (03_RESEARCH_CONTEXT.md / 07_MODELS.md). This implementation preserves
    the existing LeakyReLU(0.2) behavior unchanged; reconciling the doc/code
    mismatch is a separate action item, orthogonal to this ablation.

    RESIDUAL CONNECTIONS: controlled by `residual: bool` (default False).
    When True, each layer attempts a skip connection from its input to its
    raw conv output, applied BEFORE normalization/activation/dropout:

        residual = x
        x = conv(x, edge_index)
        if residual.shape == x.shape:
            x = x + residual

    No learnable projection is introduced; the add is a shape-guarded no-op
    when dimensions differ, per spec. See the residual-ablation change note
    for full details -- unaffected by the normalization change below.

    NORMALIZATION (this change): controlled by `normalization: str`, one of
    "none" | "batch" | "layer" | "graph" (default "batch" = identical to the
    prior BatchNorm1d-only implementation). Only the normalization module
    itself changes; its position in the forward pass (after the residual
    add, before activation and dropout) is unchanged, as is everything else
    (architecture, activation, heads, hidden dims, dropout, optimizer,
    scheduler, graph construction, pooling, classifier).

    GraphNorm requires the PyG `batch` vector (it normalizes per-graph
    within a batch, not per-feature-channel across the whole batch like
    BatchNorm1d, and not per-sample like LayerNorm), so `forward()` passes
    `batch` to the norm layer only when normalization="graph"; the other
    three variants are called as `norm(x)` exactly as before.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.2,
        residual: bool = False,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError(
                f"hidden_dims must contain at least one layer width, got {hidden_dims}"
            )

        self.num_layers = len(hidden_dims)
        self.residual = residual
        self.normalization = normalization.lower()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev_dim = input_dim
        for i, h in enumerate(hidden_dims):
            out_per_head = h
            is_last = i == len(hidden_dims) - 1
            n_heads = 1 if is_last else heads
            concat = not is_last
            self.convs.append(
                GATConv(prev_dim, out_per_head, heads=n_heads,
                        dropout=attn_dropout, concat=concat)
            )
            out_dim = out_per_head * n_heads if concat else out_per_head
            self.norms.append(_build_norm_layer(self.normalization, out_dim))
            prev_dim = out_dim
        self.classifier = nn.Linear(prev_dim, 1)
        self.dropout = dropout

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            if self.residual:
                residual = x
                x = conv(x, edge_index)
                if residual.shape == x.shape:
                    x = x + residual
            else:
                x = conv(x, edge_index)

            if self.normalization == "graph":
                x = norm(x, batch)
            else:
                x = norm(x)

            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x).view(-1)


def build_gat(input_dim: int, cfg: Dict[str, Any]) -> GAT:
    """
    Build a GAT model from a configuration dict.

    Depth is controlled by EITHER:
      - cfg["hidden_dims"]: explicit per-layer widths (existing behavior,
        used by the standard benchmark / Optuna search), OR
      - cfg["num_layers"]: an integer number of layers; hidden_dims is then
        auto-derived via _default_hidden_dims() following the baseline's
        width strategy (128 intermediate, 64 final).

    If both are present, cfg["hidden_dims"] takes precedence only when its
    length already matches cfg["num_layers"]; otherwise num_layers wins and
    hidden_dims is regenerated. This keeps the function fully backward
    compatible with every existing call site (registry.py, benchmark.py,
    tuning_graph.py) that only ever sets "hidden_dims".

    Residual connections: cfg["residual"] (bool, default False).

    Normalization: cfg["normalization"] (str, default "batch"). One of
    "none", "batch", "layer", "graph". Omitting the key reproduces the
    exact original BatchNorm1d-only implementation.

    Args:
        input_dim: Number of input node features.
        cfg: Model configuration dict. Recognized keys:
            hidden_dims (List[int], optional)
            num_layers (int, optional)
            heads (int)
            dropout (float)
            attn_dropout (float)
            residual (bool, optional, default False)
            normalization (str, optional, default "batch")

    Returns:
        Instantiated GAT model.
    """
    num_layers: Optional[int] = cfg.get("num_layers", None)
    hidden_dims: Optional[List[int]] = cfg.get("hidden_dims", None)

    if num_layers is not None:
        if hidden_dims is None or len(hidden_dims) != num_layers:
            hidden_dims = _default_hidden_dims(num_layers)
    elif hidden_dims is None:
        hidden_dims = [256, 128]  # original fallback default, unchanged

    return GAT(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        heads=cfg.get("heads", 4),
        dropout=cfg.get("dropout", 0.3),
        attn_dropout=cfg.get("attn_dropout", 0.2),
        residual=cfg.get("residual", False),
        normalization=cfg.get("normalization", "batch"),
    )