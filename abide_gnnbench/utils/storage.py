# abide_gnnbench/utils/storage.py

from __future__ import annotations
import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Any, Dict
from abide_gnnbench.config import OutputConfig


def ensure_dirs(cfg: OutputConfig) -> None:
    for d in [cfg.base_dir, cfg.models_dir, cfg.metrics_dir,
              cfg.plots_dir, cfg.explainability_dir, cfg.tuning_dir]:
        os.makedirs(d, exist_ok=True)


def save_model(model: torch.nn.Module, model_name: str, atlas: str, cfg: OutputConfig) -> str:
    path = os.path.join(cfg.models_dir, f"{model_name}_{atlas}.pt")
    torch.save(model.state_dict(), path)
    return path


def save_fold_models(fold_models, model_name: str, atlas: str, cfg: OutputConfig) -> None:
    for i, m in enumerate(fold_models):
        path = os.path.join(cfg.models_dir, f"{model_name}_{atlas}_fold{i+1}.pt")
        torch.save(m.state_dict(), path)


def save_metrics(
    fold_metrics_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    model_name: str,
    atlas: str,
    cfg: OutputConfig,
) -> None:
    base = f"{model_name}_{atlas}"
    fold_metrics_df.to_csv(os.path.join(cfg.metrics_dir, f"{base}_fold_metrics.csv"))
    oof_df.to_csv(os.path.join(cfg.metrics_dir, f"{base}_oof_predictions.csv"), index=False)
    ci_df.to_csv(os.path.join(cfg.metrics_dir, f"{base}_bootstrap_ci.csv"))


def save_tuning_results(df: pd.DataFrame, model_name: str, atlas: str, cfg: OutputConfig) -> None:
    path = os.path.join(cfg.tuning_dir, f"{model_name}_{atlas}_tuning.csv")
    df.to_csv(path, index=False)


def save_explainability(data: Dict[str, Any], model_name: str, atlas: str, cfg: OutputConfig) -> None:
    path = os.path.join(cfg.explainability_dir, f"{model_name}_{atlas}_explain.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_json(data: Dict[str, Any], name: str, cfg: OutputConfig) -> None:
    path = os.path.join(cfg.base_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
