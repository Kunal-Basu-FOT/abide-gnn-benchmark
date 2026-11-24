# abide_gnnbench/experiments/run_mlp.py

from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from abide_gnnbench.config import (
    BenchmarkConfig, MODEL_DEFAULTS, MODEL_SEARCH_SPACES,
)
from abide_gnnbench.utils.seed import set_seed, get_device
from abide_gnnbench.utils.data import load_abide
from abide_gnnbench.utils.training import train_tabular, predict_tabular
from abide_gnnbench.utils.cv_engine import run_cv_tabular
from abide_gnnbench.utils.tuning import run_optuna_tuning
from abide_gnnbench.utils.plotting import (
    plot_dataset_overview, plot_tuning_results, plot_multi_model_comparison,
)
from abide_gnnbench.utils.explainability import (
    compute_permutation_importance, plot_permutation_importance,
    compute_shap_mlp, compute_integrated_gradients, plot_integrated_gradients,
)
from abide_gnnbench.utils.storage import ensure_dirs, save_tuning_results
from abide_gnnbench.models.mlp import build_mlp
from abide_gnnbench.utils.data import make_val_split, scale_fold


MODEL_NAME = "MLP"


def run_mlp_experiment(
    atlas: str,
    cfg: BenchmarkConfig,
    run_tuning: bool = True,
    run_explainability: bool = True,
    n_tuning_trials: int = 25,
) -> dict:
    plt.style.use(cfg.plot.style)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.output)

    print(f"\n{'#'*70}")
    print(f"  {MODEL_NAME} Experiment — {atlas}")
    print(f"  Device: {device}")
    print(f"{'#'*70}")

    data_path = cfg.data.aal_path if atlas == "AAL" else cfg.data.cc200_path
    X, y, site_ids, subject_idx, fc_cols = load_abide(data_path, cfg.data)
    plot_dataset_overview(y, site_ids, MODEL_NAME, atlas, cfg.plot)

    baseline_cfg = MODEL_DEFAULTS[MODEL_NAME].copy()

    print(f"\n{'='*60}")
    print(f"  STEP 1 — Baseline {MODEL_NAME}")
    print(f"{'='*60}")
    for k, v in baseline_cfg.items():
        print(f"  {k:20s}: {v}")

    baseline_result = run_cv_tabular(
        model_name=MODEL_NAME,
        atlas=atlas,
        X=X, y=y, site_ids=site_ids, subject_idx=subject_idx,
        build_model_fn=build_mlp,
        train_fn=train_tabular,
        predict_fn=predict_tabular,
        model_cfg=baseline_cfg,
        device=device,
        cv_cfg=cfg.cv,
        output_cfg=cfg.output,
        plot_cfg=cfg.plot,
        tag="baseline",
    )

    best_cfg = baseline_cfg.copy()

    if run_tuning:
        print(f"\n{'='*60}")
        print(f"  STEP 2 — Optuna Tuning [{atlas} {MODEL_NAME}] ({n_tuning_trials} trials)")
        print(f"{'='*60}")

        tuning_df, best_params = run_optuna_tuning(
            model_name=MODEL_NAME,
            X=X, y=y,
            build_model_fn=build_mlp,
            train_fn=train_tabular,
            predict_fn=predict_tabular,
            device=device,
            cv_cfg=cfg.cv,
            search_space=MODEL_SEARCH_SPACES[MODEL_NAME],
            n_trials=n_tuning_trials,
            seed=cfg.seed,
        )
        save_tuning_results(tuning_df, MODEL_NAME, atlas, cfg.output)
        plot_tuning_results(tuning_df, MODEL_NAME, atlas, cfg.plot)

        best_cfg.update(best_params)

        print(f"\n  Best Configuration:")
        for k, v in best_cfg.items():
            print(f"    {k:20s}: {v}")

        print(f"\n{'='*60}")
        print(f"  STEP 3 — Final Tuned {MODEL_NAME}")
        print(f"{'='*60}")

        tuned_result = run_cv_tabular(
            model_name=MODEL_NAME,
            atlas=atlas,
            X=X, y=y, site_ids=site_ids, subject_idx=subject_idx,
            build_model_fn=build_mlp,
            train_fn=train_tabular,
            predict_fn=predict_tabular,
            model_cfg=best_cfg,
            device=device,
            cv_cfg=cfg.cv,
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
        print(f"  STEP 4 — Explainability")
        print(f"{'='*60}")

        print("\n[1/3] Permutation Feature Importance...")
        perm_df = compute_permutation_importance(
            fold_models=final_result["fold_models"],
            fold_test_data=final_result["fold_test_data"],
            feature_names=fc_cols,
            device=device,
            cfg=cfg.explain,
        )
        plot_permutation_importance(perm_df, MODEL_NAME, atlas, cfg.explain, cfg.plot)
        print("\nTop 20 Permutation Features:")
        print(perm_df.head(20).to_string(index=False))

        print("\n[2/3] SHAP...")
        best_fold_idx = int(final_result["fold_metrics_df"]["AUROC"].argmax())
        from sklearn.model_selection import StratifiedKFold
        skf_ex = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.seed)
        splits = list(skf_ex.split(X, y))
        train_idx_best, test_idx_best = splits[best_fold_idx]
        X_tr_b, y_tr_b, X_val_b, y_val_b = make_val_split(
            X[train_idx_best], y[train_idx_best],
            val_frac=cfg.cv.val_frac, seed=cfg.seed + best_fold_idx,
        )
        X_tr_bs, X_val_bs, X_test_bs, _ = scale_fold(X_tr_b, X_val_b, X[test_idx_best])
        best_model = final_result["fold_models"][best_fold_idx]
        best_model.eval()

        shap_df = compute_shap_mlp(
            best_model=best_model,
            X_tr_bs=X_tr_bs,
            X_test_bs=X_test_bs,
            feature_names=fc_cols,
            device=device,
            cfg=cfg.explain,
        )
        print("\nTop 20 SHAP Features:")
        print(shap_df.head(20).to_string())

        print("\n[3/3] Integrated Gradients...")
        ig_df = compute_integrated_gradients(
            best_model=best_model,
            X_test_bs=X_test_bs,
            feature_names=fc_cols,
            device=device,
            cfg=cfg.explain,
        )
        plot_integrated_gradients(ig_df, MODEL_NAME, atlas, cfg.explain, cfg.plot)
        print("\nTop 20 IG Features:")
        print(ig_df.head(20).to_string())
    else:
        perm_df = shap_df = ig_df = None

    _print_final_summary(
        atlas=atlas,
        n_subjects=len(y),
        n_features=len(fc_cols),
        cfg_used=best_cfg,
        result=final_result,
        perm_df=perm_df,
        shap_df=shap_df,
        ig_df=ig_df,
        n_splits=cfg.cv.n_splits,
    )

    return {
        "baseline_result": baseline_result,
        "final_result": final_result,
        "best_cfg": best_cfg,
        "fc_cols": fc_cols,
    }


def _print_final_summary(
    atlas, n_subjects, n_features, cfg_used, result,
    perm_df, shap_df, ig_df, n_splits,
):
    fold_metrics_df = result["fold_metrics_df"]
    ci = result["ci"]
    best_fold = int(fold_metrics_df["AUROC"].idxmax())
    worst_fold = int(fold_metrics_df["AUROC"].idxmin())

    print("\n" + "=" * 70)
    print(f"  {MODEL_NAME} — {atlas} — {n_splits}-Fold CV FINAL SUMMARY")
    print("=" * 70)
    print(f"  Subjects         : {n_subjects}")
    print(f"  FC Features      : {n_features}")
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

    if perm_df is not None:
        print("\n  Top 10 Permutation Features:")
        for i, row in perm_df.head(10).iterrows():
            print(f"    {i+1:2d}. {row['Feature']:35s}  {row['Importance']:.6f}")

    if shap_df is not None:
        print("\n  Top 10 SHAP Features:")
        for i, row in shap_df.head(10).iterrows():
            print(f"    {i+1:2d}. {row['Feature']:35s}  {row['Mean |SHAP|']:.6f}")

    if ig_df is not None:
        print("\n  Top 10 Integrated Gradients Features:")
        for i, row in ig_df.head(10).iterrows():
            print(f"    {i+1:2d}. {row['Feature']:35s}  {row['Mean |Attribution|']:.6f}")

    print("=" * 70)
    print(f"  END OF {MODEL_NAME} — {atlas}")
    print("=" * 70)