# abide_gnnbench/utils/tuning_graph.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from abide_gnnbench.config import CVConfig, GraphConfig, MODEL_SEARCH_SPACES
from abide_gnnbench.utils.seed import set_seed
from abide_gnnbench.utils.data import make_val_split, scale_fold
from abide_gnnbench.utils.graph import build_graph_dataset
from abide_gnnbench.utils.metrics import compute_metrics, METRIC_KEYS

def _sample_from_space(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key, spec in space.items():
        t = spec["type"]
        if t == "categorical":
            cfg[key] = trial.suggest_categorical(key, spec["choices"])
        elif t == "float":
            if "step" in spec:
                cfg[key] = trial.suggest_float(key, spec["low"], spec["high"], step=spec["step"])
            else:
                cfg[key] = trial.suggest_float(key, spec["low"], spec["high"])
        elif t == "loguniform":
            cfg[key] = trial.suggest_float(key, spec["low"], spec["high"], log=True)
        elif t == "int":
            cfg[key] = trial.suggest_int(key, spec["low"], spec["high"])
    return cfg


def run_optuna_tuning_graph(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_rois: int,
    build_model_fn: Callable,
    train_fn: Callable,
    predict_fn: Callable,
    device,
    cv_cfg: CVConfig,
    graph_cfg: GraphConfig,
    search_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 30,
    seed: int = 42,
    extra_fixed: Optional[Dict[str, Any]] = None,
) -> tuple:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if search_space is None:
        search_space = MODEL_SEARCH_SPACES[model_name]

    skf = StratifiedKFold(n_splits=cv_cfg.tune_folds, shuffle=True, random_state=seed)
    extra_fixed = extra_fixed or {}

    def objective(trial):
        cfg = _sample_from_space(trial, search_space)
        cfg.update(extra_fixed)
        cfg.setdefault("max_epochs", 150)
        cfg.setdefault("patience", 20)
        cfg.setdefault("batch_size", 32)

        fold_metrics_list = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            set_seed(seed + fold_idx)
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_all, y_test = y[train_idx], y[test_idx]

            X_tr, y_tr, X_val, y_val = make_val_split(
                X_train_raw, y_train_all,
                val_frac=cv_cfg.val_frac,
                seed=seed + fold_idx,
            )
            X_tr_s, X_val_s, X_test_s, _ = scale_fold(X_tr, X_val, X_test_raw)

            train_graphs = build_graph_dataset(X_tr_s, y_tr, n_rois, graph_cfg)
            val_graphs = build_graph_dataset(X_val_s, y_val, n_rois, graph_cfg)
            test_graphs = build_graph_dataset(X_test_s, y_test, n_rois, graph_cfg)

            node_feat_dim = train_graphs[0].x.shape[1]
            model = build_model_fn(input_dim=node_feat_dim, cfg=cfg)

            model, _ = train_fn(
                model, train_graphs, val_graphs,
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-4),
                batch_size=cfg.get("batch_size", 32),
                max_epochs=cfg.get("max_epochs", 150),
                patience=cfg.get("patience", 20),
                device=device,
            )
            probs, preds = predict_fn(model, test_graphs, device=device)

            if len(np.unique(y_test)) < 2:
                continue
            fold_metrics_list.append(compute_metrics(y_test, preds, probs))

        if not fold_metrics_list:
            for k in METRIC_KEYS:
                trial.set_user_attr(f"mean_{k}", 0.5 if k == "AUROC" else 0.0)
            return 0.5

        mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in METRIC_KEYS}
        for k, v in mean_metrics.items():
            trial.set_user_attr(f"mean_{k}", v)

        # ── CHANGE ONLY THIS LINE TO SWITCH OBJECTIVE ──────────────────────
        return mean_metrics["AUROC"]
        # Examples:
        # return mean_metrics["F1"]
        # return mean_metrics["Balanced_Acc"]
        # return 0.6 * mean_metrics["AUROC"] + 0.4 * mean_metrics["Balanced_Acc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)

    rename_map = {}
    for k in METRIC_KEYS:
        rename_map[f"user_attrs_mean_{k}"] = f"mean_{k}"
    trials_df.rename(columns=rename_map, inplace=True)
    # Drop the raw "value" column — mean_AUROC already comes from user_attrs
    if "value" in trials_df.columns:
        trials_df.drop(columns=["value"], inplace=True)

    print(f"\n── Graph Optuna Tuning Results — {model_name} ──")
    print(f"  Best AUROC : {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")

    return trials_df, study.best_params