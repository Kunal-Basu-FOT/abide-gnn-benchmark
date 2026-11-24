# abide_gnnbench/experiments/run_gnn.py

from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from abide_gnnbench.config import BenchmarkConfig, MODEL_DEFAULTS, MODEL_SEARCH_SPACES
from abide_gnnbench.utils.seed import set_seed, get_device
from abide_gnnbench.utils.data import load_abide
from abide_gnnbench.utils.graph import n_rois_from_atlas
from abide_gnnbench.utils.cv_engine_graph import run_cv_graph
from abide_gnnbench.utils.tuning_graph import run_optuna_tuning_graph
from abide_gnnbench.utils.explainability import compute_gnn_explainer
from abide_gnnbench.utils.storage import ensure_dirs, save_tuning_results
from abide_gnnbench.utils.plotting import plot_dataset_overview, plot_tuning_results
from abide_gnnbench.models.registry import get_model_entry

_GNN_NAMES = {"GCN", "GAT", "ChebNet", "GIN", "GraphSAGE"}


def run_gnn_experiment(
    model_name: str,
    atlas: str,
    cfg: BenchmarkConfig,
    run_tuning: bool = False,
    run_explainability: bool = True,
    n_tuning_trials: int = 30,
) -> dict:
    if model_name not in _GNN_NAMES:
        raise ValueError(f"Use run_mlp_experiment for MLP. Got: {model_name}")

    plt.style.use(cfg.plot.style)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.output)

    print(f"\n{'#'*70}")
    print(f"  {model_name} Experiment — {atlas}")
    print(f"  Device: {device}")
    print(f"{'#'*70}")

    data_path = cfg.data.aal_path if atlas == "AAL" else cfg.data.cc200_path
    X, y, site_ids, subject_idx, fc_cols = load_abide(data_path, cfg.data)
    plot_dataset_overview(y, site_ids, model_name, atlas, cfg.plot)

    n_rois = n_rois_from_atlas(atlas)
    model_entry = get_model_entry(model_name)
    baseline_cfg = MODEL_DEFAULTS[model_name].copy()

    print(f"\n{'='*60}")
    print(f"  STEP 1 — {model_name} Baseline [{atlas}]")
    print(f"{'='*60}")
    for k, v in baseline_cfg.items():
        print(f"  {k:20s}: {v}")

    baseline_result = run_cv_graph(
        model_name=model_name,
        atlas=atlas,
        n_rois=n_rois,
        X=X, y=y, site_ids=site_ids, subject_idx=subject_idx,
        build_model_fn=model_entry["build_fn"],
        train_fn=model_entry["train_fn"],
        predict_fn=model_entry["predict_fn"],
        model_cfg=baseline_cfg,
        device=device,
        cv_cfg=cfg.cv,
        graph_cfg=cfg.graph,
        output_cfg=cfg.output,
        plot_cfg=cfg.plot,
        tag="baseline",
    )

    best_cfg = baseline_cfg.copy()

    if run_tuning:
        print(f"\n{'='*60}")
        print(f"  STEP 2 — Optuna Tuning [{atlas} {model_name}] ({n_tuning_trials} trials)")
        print(f"{'='*60}")

        tuning_df, best_params = run_optuna_tuning_graph(
            model_name=model_name,
            X=X, y=y,
            n_rois=n_rois,
            build_model_fn=model_entry["build_fn"],
            train_fn=model_entry["train_fn"],
            predict_fn=model_entry["predict_fn"],
            device=device,
            cv_cfg=cfg.cv,
            graph_cfg=cfg.graph,
            search_space=MODEL_SEARCH_SPACES[model_name],
            n_trials=n_tuning_trials,
            seed=cfg.seed,
        )
        save_tuning_results(tuning_df, model_name, atlas, cfg.output)
        plot_tuning_results(tuning_df, model_name, atlas, cfg.plot)
        best_cfg.update(best_params)

        print(f"\n  Best Configuration:")
        for k, v in best_cfg.items():
            print(f"    {k:20s}: {v}")

        print(f"\n{'='*60}")
        print(f"  STEP 3 — Tuned {model_name} [{atlas}]")
        print(f"{'='*60}")

        tuned_result = run_cv_graph(
            model_name=model_name,
            atlas=atlas,
            n_rois=n_rois,
            X=X, y=y, site_ids=site_ids, subject_idx=subject_idx,
            build_model_fn=model_entry["build_fn"],
            train_fn=model_entry["train_fn"],
            predict_fn=model_entry["predict_fn"],
            model_cfg=best_cfg,
            device=device,
            cv_cfg=cfg.cv,
            graph_cfg=cfg.graph,
            output_cfg=cfg.output,
            plot_cfg=cfg.plot,
            tag="tuned",
        )

        print(f"\n── Baseline vs Tuned ──")
        for metric in ["Accuracy", "F1", "AUROC", "AUPRC"]:
            b = baseline_result["pooled_metrics"][metric]
            t = tuned_result["pooled_metrics"][metric]
            print(f"  {metric:12s}: Baseline={b:.4f}  Tuned={t:.4f}  Δ={t-b:+.4f}")

        final_result = tuned_result
    else:
        final_result = baseline_result

    if run_explainability:
        print(f"\n{'='*60}")
        print(f"  Explainability — GNNExplainer [{atlas} {model_name}]")
        print(f"{'='*60}")
        best_fold_idx = int(final_result["fold_metrics_df"]["AUROC"].argmax())
        best_model = final_result["fold_models"][best_fold_idx]
        _, _y_test, test_graphs = final_result["fold_test_data"][best_fold_idx]

        try:
            gnn_explain = compute_gnn_explainer(
                model=best_model,
                graphs=test_graphs,
                device=device,
                cfg=cfg.explain,
                n_explain=min(10, len(test_graphs)),
            )
            if "mean_node_importance" in gnn_explain:
                ni = gnn_explain["mean_node_importance"]
                top_nodes = np.argsort(ni)[::-1][:20]
                print(f"\n  Top 20 ROI Node Importances:")
                for rank, node_idx in enumerate(top_nodes):
                    print(f"    {rank+1:2d}. ROI {node_idx:3d}  {ni[node_idx]:.6f}")
            if "mean_edge_importance" in gnn_explain:
                ei = gnn_explain["mean_edge_importance"]
                print(f"\n  Mean Edge Importance: {ei.mean():.6f}  (std={ei.std():.6f})")
        except Exception as e:
            print(f"  GNNExplainer skipped: {e}")

    _print_gnn_summary(model_name, atlas, len(y), n_rois, best_cfg, final_result, cfg.cv.n_splits)

    return {
        "baseline_result": baseline_result,
        "final_result": final_result,
        "best_cfg": best_cfg,
        "fc_cols": fc_cols,
        "n_rois": n_rois,
    }


def _print_gnn_summary(
    model_name: str,
    atlas: str,
    n_subjects: int,
    n_rois: int,
    cfg_used: dict,
    result: dict,
    n_splits: int,
) -> None:
    fold_metrics_df = result["fold_metrics_df"]
    ci = result["ci"]
    best_fold = int(fold_metrics_df["AUROC"].idxmax())
    worst_fold = int(fold_metrics_df["AUROC"].idxmin())

    print("\n" + "=" * 70)
    print(f"  {model_name} — {atlas} — {n_splits}-Fold CV FINAL SUMMARY")
    print("=" * 70)
    print(f"  Subjects         : {n_subjects}")
    print(f"  ROIs             : {n_rois}")
    print(f"  Folds            : {n_splits}")
    print("\n  Configuration:")
    for k, v in cfg_used.items():
        print(f"    {k:22s}: {v}")

    metrics_order = [
        "Accuracy", "Sensitivity", "Specificity", "Precision",
        "F1", "AUROC", "AUPRC", "Balanced_Acc", "Brier", "MCC", "Cohen_Kappa",
    ]
    print("\n  Metrics with 95% Bootstrap CI:")
    for m in metrics_order:
        if m in ci:
            mean_, lo_, hi_ = ci[m]
            print(f"    {m:20s}: {mean_:.4f}  [{lo_:.4f} – {hi_:.4f}]")

    print(f"\n  Best Fold (AUROC) : Fold {best_fold}  ({fold_metrics_df.loc[best_fold, 'AUROC']:.4f})")
    print(f"  Worst Fold (AUROC): Fold {worst_fold} ({fold_metrics_df.loc[worst_fold, 'AUROC']:.4f})")
    print("=" * 70)
    print(f"  END OF {model_name} — {atlas}")
    print("=" * 70)
