# abide_gnnbench/utils/cv_engine_loso.py
"""
Leave-One-Site-Out (LOSO) cross-validation engine.

Design rationale
----------------
In the per-site k-fold approach the model sees data from the *same* site in
both training and test splits, making it easy for the model to exploit site-
specific acquisition artefacts rather than learning genuine ASD biomarkers.
LOSO is the established alternative used in most ABIDE benchmarking papers
(Nielsen et al. 2013; Heinsfeld et al. 2018; Jiang & Zhu 2020; etc.):
  • Test  fold  = all subjects from one site
  • Train fold  = all subjects from every *other* site
This gives one fold per site and a true held-out generalisation estimate.

Both tabular (MLP) and graph (GCN / GIN / GAT / …) models share this module;
pass `is_graph=True` and the extra `graph_cfg` / `n_rois` arguments to switch
to the graph branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from abide_gnnbench.config import CVConfig, GraphConfig, OutputConfig
from abide_gnnbench.utils.seed import set_seed
from abide_gnnbench.utils.data import make_val_split, scale_fold
from abide_gnnbench.utils.metrics import compute_metrics, bootstrap_ci
from abide_gnnbench.utils.storage import (
    ensure_dirs, save_fold_models, save_metrics,
)
from abide_gnnbench.utils.plotting import (
    plot_training_curves, plot_per_fold_curves, plot_core_evaluation,
    plot_probability_and_foldwise, plot_metric_summary, plot_metric_boxplots,
    plot_threshold_analysis,
)

# Graph utilities imported lazily so the tabular path has no hard dependency
# on torch-geometric.
def _build_graphs(X, y, n_rois, graph_cfg):
    from abide_gnnbench.utils.graph import build_graph_dataset
    return build_graph_dataset(X, y, n_rois, graph_cfg)


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------

def _site_eligible_loso(y_site: np.ndarray) -> Tuple[bool, str]:
    """A test site is eligible if it has at least one subject of each class."""
    n_asd = int(y_site.sum())
    n_td  = len(y_site) - n_asd
    if n_asd == 0 or n_td == 0:
        return False, f"single-class test site (ASD={n_asd}, TD={n_td})"
    return True, ""


def _train_sites_eligible(y_train: np.ndarray) -> Tuple[bool, str]:
    """Training pool (all other sites) must have both classes present."""
    n_asd = int(y_train.sum())
    n_td  = len(y_train) - n_asd
    if n_asd == 0 or n_td == 0:
        return False, f"training pool missing a class (ASD={n_asd}, TD={n_td})"
    return True, ""


# ---------------------------------------------------------------------------
# Main LOSO engine
# ---------------------------------------------------------------------------

def run_loso(
    model_name: str,
    atlas: str,
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
    output_cfg: OutputConfig,
    plot_cfg,
    # ---- graph-specific (ignored when is_graph=False) ----
    is_graph: bool = False,
    graph_cfg: Optional[GraphConfig] = None,
    n_rois: Optional[int] = None,
    # ---- misc ----
    tag: str = "loso",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Leave-One-Site-Out CV.

    Each unique site in `site_ids` is used as the test fold exactly once;
    the remaining sites form the training pool.  A small stratified validation
    split is carved out of the training pool for early stopping / model
    selection (controlled by `cv_cfg.val_frac`).

    Parameters
    ----------
    model_name, atlas, X, y, site_ids, subject_idx :
        Standard ABIDE benchmark inputs from `load_abide`.
    build_model_fn, train_fn, predict_fn :
        Model-specific callables (same interface as the k-fold engines).
    is_graph : bool
        When True, graph datasets are built from scaled FC matrices before
        training/prediction.  Requires `graph_cfg` and `n_rois`.
    tag :
        Appended to file names to distinguish LOSO runs from k-fold runs.

    Returns
    -------
    dict with keys: oof_df, fold_metrics_df, ci_df, ci, pooled_metrics,
                    fold_models, fold_histories, fold_test_data,
                    sites_run, sites_skipped,
                    y_true, y_prob, y_pred
    """
    ensure_dirs(output_cfg)
    seed = cv_cfg.seed

    unique_sites = sorted(set(site_ids))
    n_sites      = len(unique_sites)

    oof_records:       List[Dict]  = []
    fold_metrics_list: List[Dict]  = []
    fold_histories:    List[Dict]  = []
    fold_models:       List        = []
    fold_test_data:    List[Tuple] = []
    sites_run:         List[str]   = []
    sites_skipped:     List[Dict]  = []

    if verbose:
        mode = "Graph-LOSO" if is_graph else "Tabular-LOSO"
        print(f"\n{'='*60}")
        print(f"  {atlas}  {model_name}  —  {n_sites}-Site LOSO [{tag}]  ({mode})")
        print(f"{'='*60}")

    for fold_idx, test_site in enumerate(
        tqdm(unique_sites, desc=f"{model_name} LOSO", total=n_sites)
    ):
        set_seed(seed + fold_idx)

        test_mask  = site_ids == test_site
        train_mask = ~test_mask

        X_test_raw, y_test = X[test_mask], y[test_mask]
        X_train_raw, y_train_all = X[train_mask], y[train_mask]

        # --- eligibility checks ---
        ok, reason = _site_eligible_loso(y_test)
        if not ok:
            if verbose:
                print(f"  [fold {fold_idx+1}] SKIP test site {test_site}: {reason}")
            sites_skipped.append({"Site": test_site, "Reason": reason})
            continue

        ok, reason = _train_sites_eligible(y_train_all)
        if not ok:
            if verbose:
                print(f"  [fold {fold_idx+1}] SKIP test site {test_site} (train pool): {reason}")
            sites_skipped.append({"Site": test_site, "Reason": reason})
            continue

        # --- validation split (stratified, from training pool only) ---
        X_tr, y_tr, X_val, y_val = make_val_split(
            X_train_raw, y_train_all,
            val_frac=cv_cfg.val_frac,
            seed=seed + fold_idx,
        )

        # --- scaling (fit on train, apply to val + test) ---
        X_tr_s, X_val_s, X_test_s, _ = scale_fold(X_tr, X_val, X_test_raw)

        # --- build + train ---
        if is_graph:
            assert graph_cfg is not None and n_rois is not None, (
                "graph_cfg and n_rois must be provided when is_graph=True"
            )
            train_graphs = _build_graphs(X_tr_s,   y_tr,    n_rois, graph_cfg)
            val_graphs   = _build_graphs(X_val_s,  y_val,   n_rois, graph_cfg)
            test_graphs  = _build_graphs(X_test_s, y_test,  n_rois, graph_cfg)

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
            fold_test_data.append((X_test_s, y_test, test_graphs))

        else:
            model = build_model_fn(input_dim=X_tr_s.shape[1], cfg=model_cfg)
            model, history = train_fn(
                model, X_tr_s, y_tr, X_val_s, y_val,
                lr=model_cfg.get("lr", 1e-3),
                weight_decay=model_cfg.get("weight_decay", 1e-4),
                batch_size=model_cfg.get("batch_size", 32),
                max_epochs=model_cfg.get("max_epochs", 150),
                patience=model_cfg.get("patience", 20),
                device=device,
            )
            probs, preds = predict_fn(model, X_test_s, device=device)
            fold_test_data.append((X_test_s, y_test))

        # --- record OOF predictions (subject-level) ---
        test_subject_idxs = subject_idx[test_mask]
        for i in range(len(y_test)):
            oof_records.append({
                "SUBJECT_IDX": test_subject_idxs[i],
                "TRUE_LABEL":  int(y_test[i]),
                "PROB":        float(probs[i]),
                "PRED_LABEL":  int(preds[i]),
                "SITE_ID":     test_site,           # site that was left out
                "FOLD":        fold_idx + 1,
            })

        # --- per-site (per-fold) metrics ---
        m = compute_metrics(y_test, preds, probs)
        m["Fold"]    = fold_idx + 1
        m["Site"]    = test_site
        m["N_Test"]  = len(y_test)
        m["N_ASD"]   = int(y_test.sum())
        m["N_TD"]    = int((y_test == 0).sum())
        fold_metrics_list.append(m)
        fold_histories.append(history)
        fold_models.append(model)
        sites_run.append(test_site)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    if not oof_records:
        raise RuntimeError(
            "LOSO produced no predictions — all sites were skipped. "
            "Check site eligibility or dataset integrity."
        )

    oof_df          = pd.DataFrame(oof_records)
    fold_metrics_df = (
        pd.DataFrame(fold_metrics_list)
        .set_index("Fold")
    )

    if verbose:
        print(f"\nPer-Site Metrics ({len(sites_run)} sites ran, {len(sites_skipped)} skipped):")
        display_cols = ["Site", "N_Test", "Accuracy", "F1", "AUROC", "Balanced_Acc"]
        print(fold_metrics_df[[c for c in display_cols if c in fold_metrics_df.columns]]
              .round(4).to_string())

    y_true = oof_df["TRUE_LABEL"].values
    y_prob = oof_df["PROB"].values
    y_pred = oof_df["PRED_LABEL"].values

    pooled = compute_metrics(y_true, y_pred, y_prob)

    if verbose:
        print(f"\nPooled LOSO Metrics (all {len(y_true)} subjects):")
        for k, v in pooled.items():
            print(f"  {k:20s}: {v:.4f}")
        print(f"\nComputing Bootstrap CIs (2000 resamples)...")

    ci = bootstrap_ci(y_true, y_prob, n_boot=2000, seed=seed)

    ci_rows = []
    for k, (mean, lo, hi) in ci.items():
        ci_rows.append({
            "Metric": k, "Mean": mean,
            "95% CI Lower": lo, "95% CI Upper": hi,
            "95% CI": f"{mean:.4f} [{lo:.4f}–{hi:.4f}]",
        })
    ci_df = pd.DataFrame(ci_rows).set_index("Metric")

    if verbose:
        print("\nBootstrap Confidence Intervals:")
        print(ci_df[["Mean", "95% CI Lower", "95% CI Upper", "95% CI"]].round(4).to_string())

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    atlas_tag = f"{atlas}_{tag}"
    save_fold_models(fold_models, model_name, atlas_tag, output_cfg)
    save_metrics(fold_metrics_df, oof_df, ci_df, model_name, atlas_tag, output_cfg)

    # ------------------------------------------------------------------
    # Plots (reuse existing plotting utilities; fold ≡ site here)
    # ------------------------------------------------------------------
    plot_training_curves(fold_histories, model_name, atlas, plot_cfg)
    plot_per_fold_curves(fold_histories, model_name, atlas, plot_cfg)
    plot_core_evaluation(y_true, y_prob, y_pred, model_name, atlas, plot_cfg, tag=tag)
    plot_probability_and_foldwise(
        y_true, y_prob, fold_metrics_df, model_name, atlas, plot_cfg, tag=tag
    )
    plot_metric_summary(ci, model_name, atlas, plot_cfg, tag=tag)
    plot_metric_boxplots(fold_metrics_df, model_name, atlas, plot_cfg, tag=tag)
    plot_threshold_analysis(y_true, y_prob, model_name, atlas, plot_cfg, tag=tag)

    return {
        "oof_df":           oof_df,
        "fold_metrics_df":  fold_metrics_df,
        "ci_df":            ci_df,
        "ci":               ci,
        "pooled_metrics":   pooled,
        "fold_models":      fold_models,
        "fold_histories":   fold_histories,
        "fold_test_data":   fold_test_data,
        "sites_run":        sites_run,
        "sites_skipped":    sites_skipped,
        "y_true":           y_true,
        "y_prob":           y_prob,
        "y_pred":           y_pred,
    }
