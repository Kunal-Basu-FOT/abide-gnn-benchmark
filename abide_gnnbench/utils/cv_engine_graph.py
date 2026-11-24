# abide_gnnbench/utils/cv_engine_graph.py

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import Callable, Dict, List, Optional, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from abide_gnnbench.config import CVConfig, OutputConfig, GraphConfig
from abide_gnnbench.utils.seed import set_seed
from abide_gnnbench.utils.data import make_val_split, scale_fold
from abide_gnnbench.utils.graph import build_graph_dataset
from abide_gnnbench.utils.metrics import compute_metrics, bootstrap_ci
from abide_gnnbench.utils.storage import (
    ensure_dirs, save_fold_models, save_metrics,
)
from abide_gnnbench.utils.plotting import (
    plot_training_curves, plot_per_fold_curves, plot_core_evaluation,
    plot_probability_and_foldwise, plot_metric_summary, plot_metric_boxplots,
    plot_threshold_analysis, plot_fold_class_distribution,
)


def run_cv_graph(
    model_name: str,
    atlas: str,
    n_rois: int,
    X: np.ndarray,
    y: np.ndarray,
    site_ids: np.ndarray,
    subject_idx: np.ndarray,
    build_model_fn: Callable,
    train_fn: Callable,
    predict_fn: Callable,
    model_cfg: Dict[str, Any],
    device: torch.device,
    cv_cfg: CVConfig,
    graph_cfg: GraphConfig,
    output_cfg: OutputConfig,
    plot_cfg,
    tag: str = "baseline",
    verbose: bool = True,
) -> Dict[str, Any]:
    ensure_dirs(output_cfg)
    seed = cv_cfg.seed

    skf = StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=True, random_state=seed)

    oof_records: List[Dict] = []
    fold_metrics_list: List[Dict] = []
    fold_histories: List[Dict] = []
    fold_models: List = []
    fold_test_data: List[Tuple] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {atlas} {model_name} — {cv_cfg.n_splits}-Fold CV [{tag}]  (Graph)")
        print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(skf.split(X, y), total=cv_cfg.n_splits, desc=f"{model_name} Folds")
    ):
        set_seed(seed + fold_idx)

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_all, y_test = y[train_idx], y[test_idx]

        X_tr, y_tr, X_val, y_val = make_val_split(
            X_train_raw, y_train_all,
            val_frac=cv_cfg.val_frac, seed=seed + fold_idx,
        )
        X_tr_s, X_val_s, X_test_s, _ = scale_fold(X_tr, X_val, X_test_raw)

        train_graphs = build_graph_dataset(X_tr_s, y_tr, n_rois, graph_cfg)
        val_graphs = build_graph_dataset(X_val_s, y_val, n_rois, graph_cfg)
        test_graphs = build_graph_dataset(X_test_s, y_test, n_rois, graph_cfg)

        node_feat_dim = train_graphs[0].x.shape[1]
        model = build_model_fn(input_dim=node_feat_dim, cfg=model_cfg)
        model, history = train_fn(
            model, train_graphs, val_graphs,
            lr=model_cfg.get("lr", 1e-3),
            weight_decay=model_cfg.get("weight_decay", 1e-4),
            batch_size=model_cfg.get("batch_size", 32),
            max_epochs=model_cfg.get("max_epochs", 150),
            patience=model_cfg.get("patience", 20),
            device=device,
        )

        probs, preds = predict_fn(model, test_graphs, device=device)

        for i, idx in enumerate(test_idx):
            oof_records.append({
                "SUBJECT_IDX": subject_idx[idx],
                "TRUE_LABEL": int(y_test[i]),
                "PROB": float(probs[i]),
                "PRED_LABEL": int(preds[i]),
                "FOLD": fold_idx + 1,
                "SITE_ID": site_ids[idx],
            })

        m = compute_metrics(y_test, preds, probs)
        m["Fold"] = fold_idx + 1
        fold_metrics_list.append(m)
        fold_histories.append(history)
        fold_models.append(model)
        fold_test_data.append((X_test_s, y_test, test_graphs))

    oof_df = pd.DataFrame(oof_records)
    fold_metrics_df = pd.DataFrame(fold_metrics_list).set_index("Fold")

    if verbose:
        print(f"\nPer-Fold Metrics:")
        print(fold_metrics_df.round(4).to_string())

    y_true = oof_df["TRUE_LABEL"].values
    y_prob = oof_df["PROB"].values
    y_pred = oof_df["PRED_LABEL"].values

    pooled = compute_metrics(y_true, y_pred, y_prob)

    if verbose:
        print(f"\nPooled OOF Metrics:")
        for k, v in pooled.items():
            print(f"  {k:20s}: {v:.4f}")
        print(f"\nComputing Bootstrap CIs...")

    ci = bootstrap_ci(y_true, y_prob, n_boot=2000, seed=seed)

    ci_rows = []
    for k, (mean, lo, hi) in ci.items():
        ci_rows.append({
            "Metric": k, "Mean": mean,
            "95% CI Lower": lo, "95% CI Upper": hi,
            "95% CI": f"{mean:.4f} [{lo:.4f}–{hi:.4f}]",
        })
    ci_df = pd.DataFrame(ci_rows).set_index("Metric")

    save_fold_models(fold_models, model_name, f"{atlas}_{tag}", output_cfg)
    save_metrics(fold_metrics_df, oof_df, ci_df, model_name, f"{atlas}_{tag}", output_cfg)

    plot_training_curves(fold_histories, model_name, atlas, plot_cfg)
    plot_per_fold_curves(fold_histories, model_name, atlas, plot_cfg)
    plot_core_evaluation(y_true, y_prob, y_pred, model_name, atlas, plot_cfg, tag=tag)
    plot_probability_and_foldwise(y_true, y_prob, fold_metrics_df, model_name, atlas, plot_cfg, tag=tag)
    plot_metric_summary(ci, model_name, atlas, plot_cfg, tag=tag)
    plot_metric_boxplots(fold_metrics_df, model_name, atlas, plot_cfg, tag=tag)
    plot_threshold_analysis(y_true, y_prob, model_name, atlas, plot_cfg, tag=tag)
    plot_fold_class_distribution(skf, X, y, cv_cfg.n_splits, model_name, atlas, plot_cfg)

    return {
        "oof_df": oof_df,
        "fold_metrics_df": fold_metrics_df,
        "ci_df": ci_df,
        "ci": ci,
        "pooled_metrics": pooled,
        "fold_models": fold_models,
        "fold_histories": fold_histories,
        "fold_test_data": fold_test_data,
        "skf": skf,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
