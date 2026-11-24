# abide_gnnbench/utils/explainability.py

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple, Union
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance as sk_permutation_importance

from abide_gnnbench.config import ExplainConfig, PlotConfig
from abide_gnnbench.utils.plotting import _save_fig


# ── sklearn-compatible wrapper for PyTorch MLP models ─────────────────────────

class TorchSklearnWrapper:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X.astype(np.float32)).to(self.device)
            prob = torch.sigmoid(self.model(t)).cpu().numpy().ravel()
        return np.column_stack([1 - prob, prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        prob = self.predict_proba(X)[:, 1]
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, prob)


def _roc_auc_scorer(estimator, X, y):
    prob = estimator.predict_proba(X)[:, 1]
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, prob)


def _unpack_tabular_fold(entry: Union[tuple, list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handles both tabular (X, y) 2-tuples and graph (X, y, graphs) 3-tuples.
    Returns only (X_test_s, y_test) for permutation importance.
    """
    if len(entry) == 2:
        return entry[0], entry[1]
    return entry[0], entry[1]


# ── Permutation Feature Importance ────────────────────────────────────────────

def compute_permutation_importance(
    fold_models: List[nn.Module],
    fold_test_data: List[tuple],
    feature_names: List[str],
    device: torch.device,
    cfg: ExplainConfig,
) -> pd.DataFrame:
    perm_imp_all = np.zeros(len(feature_names))
    valid_folds = 0

    for fold_idx, (model, entry) in enumerate(zip(fold_models, fold_test_data)):
        X_test_s, y_test = _unpack_tabular_fold(entry)
        if len(np.unique(y_test)) < 2:
            print(f"  Skipping Fold {fold_idx + 1}: single class in test set.")
            continue
        wrapper = TorchSklearnWrapper(model, device)
        result = sk_permutation_importance(
            estimator=wrapper,
            X=X_test_s,
            y=y_test,
            n_repeats=cfg.perm_n_repeats,
            random_state=42 + fold_idx,
            scoring=_roc_auc_scorer,
            n_jobs=1,
        )
        perm_imp_all += result.importances_mean
        valid_folds += 1

    if valid_folds == 0:
        raise ValueError("No valid folds for permutation importance.")

    perm_imp_all /= valid_folds
    df = pd.DataFrame({"Feature": feature_names, "Importance": perm_imp_all})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def plot_permutation_importance(
    perm_df: pd.DataFrame,
    model_name: str,
    atlas: str,
    cfg: ExplainConfig,
    plot_cfg: PlotConfig,
) -> None:
    plt.style.use(plot_cfg.style)
    top_k = perm_df.head(cfg.top_k_features)
    fig, ax = plt.subplots(figsize=plot_cfg.figsize_square)
    ax.barh(top_k["Feature"][::-1], top_k["Importance"][::-1], color="#3498db", edgecolor="black")
    ax.set_xlabel("Mean Decrease in AUROC")
    ax.set_title(
        f"Top {cfg.top_k_features} Permutation Important FC Features — {atlas} {model_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_perm_importance", plot_cfg)
    plt.show()


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap_mlp(
    best_model: nn.Module,
    X_tr_bs: np.ndarray,
    X_test_bs: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    cfg: ExplainConfig,
) -> pd.DataFrame:
    import shap

    bg_size = min(cfg.shap_bg_size, len(X_tr_bs))
    bg_idx = np.random.choice(len(X_tr_bs), bg_size, replace=False)
    background_data = torch.tensor(X_tr_bs[bg_idx]).float().to(device)
    explain_size = min(cfg.shap_explain_size, len(X_test_bs))
    explain_data = torch.tensor(X_test_bs[:explain_size]).float().to(device)

    try:
        explainer = shap.GradientExplainer(best_model, background_data)
        shap_values = explainer.shap_values(explain_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        if shap_values.ndim > 2:
            shap_values = shap_values.squeeze()
    except Exception as e:
        print(f"  GradientExplainer failed ({e}), falling back to KernelExplainer...")
        best_model.eval()

        def model_predict_np(X_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(X_np.astype(np.float32)).to(device)
                return torch.sigmoid(best_model(t)).cpu().numpy()

        bg_np = X_tr_bs[bg_idx]
        explainer_k = shap.KernelExplainer(model_predict_np, bg_np[:30])
        shap_values = explainer_k.shap_values(X_test_bs[:20], nsamples=100)

    explain_np = X_test_bs[:shap_values.shape[0]]

    plt.figure()
    shap.summary_plot(shap_values, explain_np, feature_names=feature_names,
                      show=False, max_display=cfg.top_k_features)
    plt.title(f"SHAP Summary Plot — Top {cfg.top_k_features} Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, explain_np, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=cfg.top_k_features)
    plt.title(f"SHAP Mean |SHAP Value| — Top {cfg.top_k_features} Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs_shap})
    df = df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    return df


# ── Integrated Gradients (Captum) ─────────────────────────────────────────────

class CaptumWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out.unsqueeze(-1) if out.dim() == 1 else out


def compute_integrated_gradients(
    best_model: nn.Module,
    X_test_bs: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    cfg: ExplainConfig,
) -> pd.DataFrame:
    from captum.attr import IntegratedGradients

    captum_model = CaptumWrapper(best_model).to(device)
    captum_model.eval()

    ig = IntegratedGradients(captum_model)
    explain_size = min(cfg.shap_explain_size, len(X_test_bs))
    input_tensor = torch.tensor(X_test_bs[:explain_size]).float().to(device)
    input_tensor.requires_grad = True
    baseline = torch.zeros_like(input_tensor).to(device)

    attributions, _ = ig.attribute(
        input_tensor, baseline, target=0,
        n_steps=cfg.ig_n_steps, return_convergence_delta=True,
    )
    attr_np = attributions.detach().cpu().numpy()
    mean_abs_attr = np.abs(attr_np).mean(axis=0)

    df = pd.DataFrame({"Feature": feature_names, "Mean |Attribution|": mean_abs_attr})
    df = df.sort_values("Mean |Attribution|", ascending=False).reset_index(drop=True)
    return df


def plot_integrated_gradients(
    ig_df: pd.DataFrame,
    model_name: str,
    atlas: str,
    cfg: ExplainConfig,
    plot_cfg: PlotConfig,
) -> None:
    plt.style.use(plot_cfg.style)
    top_k = ig_df.head(cfg.top_k_features)
    fig, ax = plt.subplots(figsize=plot_cfg.figsize_square)
    ax.barh(top_k["Feature"][::-1], top_k["Mean |Attribution|"][::-1],
            color="#e67e22", edgecolor="black")
    ax.set_xlabel("Mean |Integrated Gradient Attribution|")
    ax.set_title(
        f"Top {cfg.top_k_features} Features — Integrated Gradients — {atlas} {model_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_ig", plot_cfg)
    plt.show()


# ── GNN Explainability (GNNExplainer) ─────────────────────────────────────────

def compute_gnn_explainer(
    model: nn.Module,
    graphs,
    device: torch.device,
    cfg: ExplainConfig,
    n_explain: int = 10,
) -> Dict[str, np.ndarray]:
    from torch_geometric.explain import Explainer, GNNExplainer

    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=cfg.gnn_explain_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    node_importances: List[np.ndarray] = []
    edge_importances: List[np.ndarray] = []

    for g in graphs[:n_explain]:
        g = g.to(device)
        try:
            explanation = explainer(g.x, g.edge_index, edge_attr=g.edge_attr, batch=None)
            if explanation.node_mask is not None:
                node_importances.append(explanation.node_mask.detach().cpu().numpy())
            if explanation.edge_mask is not None:
                edge_importances.append(explanation.edge_mask.detach().cpu().numpy())
        except Exception as e:
            print(f"  GNNExplainer skipped one graph: {e}")
            continue

    results: Dict[str, np.ndarray] = {}
    if node_importances:
        node_arrays = [
            n.mean(axis=1) if n.ndim > 1 else n for n in node_importances
        ]
        min_len = min(a.shape[0] for a in node_arrays)
        node_arrays = [a[:min_len] for a in node_arrays]
        results["mean_node_importance"] = np.mean(node_arrays, axis=0)

    if edge_importances:
        min_len = min(e.shape[0] for e in edge_importances)
        edge_arrays = [e[:min_len] for e in edge_importances]
        results["mean_edge_importance"] = np.mean(edge_arrays, axis=0)

    return results


# ── GNN Node Importance Plot ──────────────────────────────────────────────────

def plot_gnn_node_importance(
    node_importance: np.ndarray,
    model_name: str,
    atlas: str,
    cfg: ExplainConfig,
    plot_cfg: PlotConfig,
    top_k: Optional[int] = None,
) -> None:
    if top_k is None:
        top_k = cfg.top_k_features
    top_k = min(top_k, len(node_importance))
    top_idx = np.argsort(node_importance)[::-1][:top_k]
    top_vals = node_importance[top_idx]
    labels = [f"ROI {i}" for i in top_idx]

    plt.style.use(plot_cfg.style)
    fig, ax = plt.subplots(figsize=plot_cfg.figsize_square)
    ax.barh(labels[::-1], top_vals[::-1], color="#9b59b6", edgecolor="black")
    ax.set_xlabel("Mean Node Importance (GNNExplainer)")
    ax.set_title(
        f"Top {top_k} ROI Node Importances — {atlas} {model_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_node_importance", plot_cfg)
    plt.show()
