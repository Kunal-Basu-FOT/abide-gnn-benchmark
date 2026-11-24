# abide_gnnbench/utils/graph.py

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Optional
from abide_gnnbench.config import GraphConfig


def fc_to_edge_index_and_weight(
    fc_matrix: np.ndarray,
    cfg: GraphConfig,
) -> tuple:
    n = fc_matrix.shape[0]
    abs_fc = np.abs(fc_matrix)

    if cfg.threshold_method == "percentile":
        thresh = np.percentile(abs_fc, cfg.threshold_value)
    else:
        thresh = cfg.threshold_value

    mask = abs_fc >= thresh
    if cfg.self_loops:
        np.fill_diagonal(mask, True)

    rows, cols = np.where(mask)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)

    if cfg.edge_weight:
        weights = fc_matrix[rows, cols]
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    else:
        edge_attr = None

    return edge_index, edge_attr


def build_graph_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_rois: int,
    cfg: GraphConfig,
    node_features: Optional[np.ndarray] = None,
) -> List[Data]:
    n_subjects = X.shape[0]
    fc_dim = X.shape[1]
    assert fc_dim == n_rois * (n_rois - 1) // 2 or fc_dim == n_rois * n_rois, \
        f"FC dim {fc_dim} inconsistent with {n_rois} ROIs"

    graphs = []
    for i in range(n_subjects):
        fc_flat = X[i]
        if fc_dim == n_rois * (n_rois - 1) // 2:
            fc_mat = _upper_tri_to_symmetric(fc_flat, n_rois)
        else:
            fc_mat = fc_flat.reshape(n_rois, n_rois)

        edge_index, edge_attr = fc_to_edge_index_and_weight(fc_mat, cfg)

        if node_features is not None:
            x_node = torch.tensor(node_features[i], dtype=torch.float32)
        else:
            x_node = torch.tensor(fc_mat, dtype=torch.float32)

        label = torch.tensor(y[i], dtype=torch.float32)
        data = Data(x=x_node, edge_index=edge_index, edge_attr=edge_attr, y=label)
        graphs.append(data)

    return graphs


def _upper_tri_to_symmetric(vec: np.ndarray, n: int) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float32)
    idx = np.triu_indices(n, k=1)
    mat[idx] = vec
    mat = mat + mat.T
    return mat


def n_rois_from_atlas(atlas: str) -> int:
    mapping = {"AAL": 116, "CC200": 200, "CC400": 392}
    if atlas not in mapping:
        raise ValueError(f"Unknown atlas: {atlas}. Supported: {list(mapping.keys())}")
    return mapping[atlas]
