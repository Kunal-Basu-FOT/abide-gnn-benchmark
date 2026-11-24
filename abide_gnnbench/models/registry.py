# abide_gnnbench/models/registry.py

from __future__ import annotations
from typing import Dict, Any, Callable, Tuple

from abide_gnnbench.models.mlp import build_mlp
from abide_gnnbench.models.gcn import build_gcn
from abide_gnnbench.models.gat import build_gat
from abide_gnnbench.models.chebnet import build_chebnet
from abide_gnnbench.models.gin import build_gin
from abide_gnnbench.models.graphsage import build_graphsage

from abide_gnnbench.utils.training import (
    train_tabular, predict_tabular,
    train_graph, predict_graph,
)

# ── Model type ────────────────────────────────────────────────────────────────
# "tabular" → uses flat FC feature vector (MLP)
# "graph"   → uses PyG Data objects (all GNNs)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "MLP": {
        "build_fn": build_mlp,
        "train_fn": train_tabular,
        "predict_fn": predict_tabular,
        "type": "tabular",
    },
    "GCN": {
        "build_fn": build_gcn,
        "train_fn": train_graph,
        "predict_fn": predict_graph,
        "type": "graph",
    },
    "GAT": {
        "build_fn": build_gat,
        "train_fn": train_graph,
        "predict_fn": predict_graph,
        "type": "graph",
    },
    "ChebNet": {
        "build_fn": build_chebnet,
        "train_fn": train_graph,
        "predict_fn": predict_graph,
        "type": "graph",
    },
    "GIN": {
        "build_fn": build_gin,
        "train_fn": train_graph,
        "predict_fn": predict_graph,
        "type": "graph",
    },
    "GraphSAGE": {
        "build_fn": build_graphsage,
        "train_fn": train_graph,
        "predict_fn": predict_graph,
        "type": "graph",
    },
}


def get_model_entry(model_name: str) -> Dict[str, Any]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def is_graph_model(model_name: str) -> bool:
    return MODEL_REGISTRY[model_name]["type"] == "graph"
