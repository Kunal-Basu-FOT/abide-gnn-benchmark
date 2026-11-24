# abide_gnnbench/utils/tuning.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score

from abide_gnnbench.config import CVConfig, MODEL_SEARCH_SPACES
from abide_gnnbench.utils.seed import set_seed
from abide_gnnbench.utils.data import make_val_split, scale_fold

def _sample_from_space(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    import optuna
    cfg = {}
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


def run_optuna_tuning(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    build_model_fn: Callable,
    train_fn: Callable,
    predict_fn: Callable,
    device,
    cv_cfg: CVConfig,
    search_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 50,
    seed: int = 42,
    extra_fixed: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
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

        fold_aurocs = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            set_seed(seed + fold_idx)
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_all, y_test = y[train_idx], y[test_idx]
            X_tr, y_tr, X_val, y_val = make_val_split(
                X_train_raw, y_train_all,
                val_frac=cv_cfg.val_frac, seed=seed + fold_idx,
            )
            X_tr_s, X_val_s, X_test_s, _ = scale_fold(X_tr, X_val, X_test_raw)

            model = build_model_fn(input_dim=X_tr_s.shape[1], cfg=cfg)
            model, _ = train_fn(
                model, X_tr_s, y_tr, X_val_s, y_val,
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-4),
                batch_size=cfg.get("batch_size", 32),
                max_epochs=cfg.get("max_epochs", 150),
                patience=cfg.get("patience", 20),
                device=device,
            )
            probs, preds = predict_fn(model, X_test_s, device=device)
            if len(np.unique(y_test)) < 2:
                continue
            fold_aurocs.append(roc_auc_score(y_test, probs))

        return np.mean(fold_aurocs) if fold_aurocs else 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df.sort_values("value", ascending=False).reset_index(drop=True)
    trials_df.rename(columns={"value": "mean_AUROC"}, inplace=True)

    print(f"\n── Optuna Tuning Results — {model_name} ──")
    print(f"  Best AUROC : {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")

    return trials_df, study.best_params


def run_grid_search(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    build_model_fn: Callable,
    train_fn: Callable,
    predict_fn: Callable,
    device,
    hp_grid: list,
    cv_cfg: CVConfig,
    seed: int = 42,
) -> pd.DataFrame:
    from tqdm import tqdm

    skf = StratifiedKFold(n_splits=cv_cfg.tune_folds, shuffle=True, random_state=seed)
    results = []

    for cfg_idx, cfg in enumerate(tqdm(hp_grid, desc=f"{model_name} Grid Search")):
        set_seed(seed)
        full_cfg = dict(max_epochs=150, patience=20, batch_size=32)
        full_cfg.update(cfg)

        fold_aurocs, fold_accs, fold_f1s, fold_auprcs = [], [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            set_seed(seed + fold_idx)
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_all, y_test = y[train_idx], y[test_idx]
            X_tr, y_tr, X_val, y_val = make_val_split(
                X_train_raw, y_train_all,
                val_frac=cv_cfg.val_frac, seed=seed + fold_idx,
            )
            X_tr_s, X_val_s, X_test_s, _ = scale_fold(X_tr, X_val, X_test_raw)

            model = build_model_fn(input_dim=X_tr_s.shape[1], cfg=full_cfg)
            model, _ = train_fn(
                model, X_tr_s, y_tr, X_val_s, y_val,
                lr=full_cfg.get("lr", 1e-3),
                weight_decay=full_cfg.get("weight_decay", 1e-4),
                batch_size=full_cfg.get("batch_size", 32),
                max_epochs=full_cfg.get("max_epochs", 150),
                patience=full_cfg.get("patience", 20),
                device=device,
            )
            probs, preds = predict_fn(model, X_test_s, device=device)
            if len(np.unique(y_test)) < 2:
                continue
            fold_aurocs.append(roc_auc_score(y_test, probs))
            fold_accs.append(accuracy_score(y_test, preds))
            fold_f1s.append(f1_score(y_test, preds, zero_division=0))
            fold_auprcs.append(average_precision_score(y_test, probs))

        row = {k: str(v) if isinstance(v, list) else v for k, v in cfg.items()}
        row["mean_AUROC"] = np.mean(fold_aurocs) if fold_aurocs else 0.5
        row["std_AUROC"] = np.std(fold_aurocs) if fold_aurocs else 0.0
        row["mean_Accuracy"] = np.mean(fold_accs) if fold_accs else 0.0
        row["mean_F1"] = np.mean(fold_f1s) if fold_f1s else 0.0
        row["mean_AUPRC"] = np.mean(fold_auprcs) if fold_auprcs else 0.0
        results.append(row)

    df = pd.DataFrame(results).sort_values(
        ["mean_AUROC", "mean_F1"], ascending=False,
    ).reset_index(drop=True)
    return df
