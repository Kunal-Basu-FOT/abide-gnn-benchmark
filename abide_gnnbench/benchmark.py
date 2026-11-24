# abide_gnnbench/benchmark.py

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, List, Optional, Tuple

from abide_gnnbench.config import BenchmarkConfig, MODEL_DEFAULTS
from abide_gnnbench.utils.seed import set_seed, get_device
from abide_gnnbench.utils.storage import ensure_dirs, save_json
from abide_gnnbench.utils.plotting import plot_multi_model_comparison
from abide_gnnbench.models.registry import is_graph_model
from abide_gnnbench.experiments.run_mlp import run_mlp_experiment
from abide_gnnbench.experiments.run_gnn import run_gnn_experiment

ALL_MODELS = ["MLP", "GCN", "ChebNet", "GIN", "GraphSAGE", "GAT"]
ALL_ATLASES = ["AAL", "CC200"]

SUMMARY_METRICS = [
    "Accuracy", "Sensitivity", "Specificity", "Precision",
    "F1", "AUROC", "AUPRC", "Balanced_Acc", "Brier", "MCC", "Cohen_Kappa",
]


def run_full_benchmark(
    cfg: BenchmarkConfig,
    models: Optional[List[str]] = None,
    atlases: Optional[List[str]] = None,
    run_tuning: bool = False,
    run_explainability: bool = True,
    n_tuning_trials: int = 30,
) -> Dict[str, Any]:
    models = models or ALL_MODELS
    atlases = atlases or ALL_ATLASES

    set_seed(cfg.seed)
    ensure_dirs(cfg.output)

    all_results: Dict[str, Dict[str, Any]] = {}
    failed: List[Tuple[str, str]] = []

    print("\n" + "#" * 70)
    print("  ABIDE-GNNBench — Full Benchmark")
    print(f"  Models : {models}")
    print(f"  Atlases: {atlases}")
    print("#" * 70)

    for atlas in atlases:
        for model_name in models:
            key = f"{model_name}_{atlas}"
            print(f"\n{'='*70}")
            print(f"  Running: {model_name} — {atlas}")
            print(f"{'='*70}")
            try:
                if model_name == "MLP":
                    exp_result = run_mlp_experiment(
                        atlas=atlas,
                        cfg=cfg,
                        run_tuning=run_tuning,
                        run_explainability=run_explainability,
                    )
                    final_result = exp_result["final_result"]
                else:
                    exp_result = run_gnn_experiment(
                        model_name=model_name,
                        atlas=atlas,
                        cfg=cfg,
                        run_tuning=run_tuning,
                        run_explainability=run_explainability,
                        n_tuning_trials=n_tuning_trials,
                    )
                    final_result = exp_result["final_result"]

                all_results[key] = {
                    "model": model_name,
                    "atlas": atlas,
                    "pooled_metrics": final_result["pooled_metrics"],
                    "ci": final_result["ci"],
                    "fold_metrics_df": final_result["fold_metrics_df"],
                    "oof_df": final_result["oof_df"],
                    "ci_df": final_result["ci_df"],
                }
            except Exception as e:
                print(f"\n  ERROR — {model_name} {atlas}: {e}")
                failed.append((model_name, atlas))

    if failed:
        print(f"\n  Failed runs: {failed}")

    _save_benchmark_summary(all_results, cfg)
    _print_benchmark_table(all_results, models, atlases)
    _generate_comparison_plots(all_results, atlases, cfg)

    if failed:
        raise RuntimeError(f"Benchmark failed for {len(failed)} model/atlas runs: {failed}")

    return all_results


def _save_benchmark_summary(
    all_results: Dict[str, Any],
    cfg: BenchmarkConfig,
) -> None:
    rows = []
    for key, res in all_results.items():
        row = {"Model": res["model"], "Atlas": res["atlas"]}
        pm = res["pooled_metrics"]
        ci = res["ci"]
        for m in SUMMARY_METRICS:
            if m in pm:
                row[m] = round(pm[m], 4)
            if m in ci:
                mean_, lo_, hi_ = ci[m]
                row[f"{m}_CI"] = f"{mean_:.4f} [{lo_:.4f}–{hi_:.4f}]"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(cfg.output.metrics_dir, "benchmark_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Benchmark summary saved: {out_path}")

    json_safe = {}
    for key, res in all_results.items():
        json_safe[key] = {
            "model": res["model"],
            "atlas": res["atlas"],
            "pooled_metrics": {k: float(v) for k, v in res["pooled_metrics"].items()},
            "ci": {
                k: [float(v) for v in tpl]
                for k, tpl in res["ci"].items()
            },
        }
    save_json(json_safe, "benchmark_results", cfg.output)


def _print_benchmark_table(
    all_results: Dict[str, Any],
    models: List[str],
    atlases: List[str],
) -> None:
    print("\n" + "=" * 90)
    print("  BENCHMARK SUMMARY TABLE")
    print("=" * 90)

    display_metrics = ["Accuracy", "F1", "AUROC", "AUPRC", "Balanced_Acc", "Sensitivity", "Specificity"]

    for atlas in atlases:
        print(f"\n  Atlas: {atlas}")
        header = f"  {'Model':12s}" + "".join(f"  {m:13s}" for m in display_metrics)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model_name in models:
            key = f"{model_name}_{atlas}"
            if key not in all_results:
                row = f"  {model_name:12s}" + "  " + "FAILED"
            else:
                pm = all_results[key]["pooled_metrics"]
                vals = "".join(
                    f"  {pm.get(m, float('nan')):13.4f}" for m in display_metrics
                )
                row = f"  {model_name:12s}{vals}"
            print(row)

    print("\n" + "=" * 90)


def _generate_comparison_plots(
    all_results: Dict[str, Any],
    atlases: List[str],
    cfg: BenchmarkConfig,
) -> None:
    for atlas in atlases:
        atlas_results: Dict[str, Dict[str, float]] = {}
        for key, res in all_results.items():
            if res["atlas"] == atlas:
                atlas_results[res["model"]] = res["pooled_metrics"]

        if len(atlas_results) >= 2:
            plot_multi_model_comparison(atlas_results, atlas, cfg.plot)

    _plot_cross_atlas_comparison(all_results, atlases, cfg)


def _plot_cross_atlas_comparison(
    all_results: Dict[str, Any],
    atlases: List[str],
    cfg: BenchmarkConfig,
) -> None:
    if len(atlases) < 2:
        return

    plt.style.use(cfg.plot.style)
    compare_metric = "AUROC"
    models_seen = list({res["model"] for res in all_results.values()})
    models_seen_sorted = [m for m in ALL_MODELS if m in models_seen]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(models_seen_sorted))
    width = 0.35
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22"]

    for i, atlas in enumerate(atlases):
        vals = []
        for model_name in models_seen_sorted:
            key = f"{model_name}_{atlas}"
            if key in all_results:
                vals.append(all_results[key]["pooled_metrics"].get(compare_metric, 0.0))
            else:
                vals.append(0.0)
        ax.bar(
            x + i * width, vals, width,
            label=atlas, color=palette[i % len(palette)], edgecolor="black",
        )
        for xi, val in zip(x + i * width, vals):
            ax.text(xi, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(atlases) - 1) / 2)
    ax.set_xticklabels(models_seen_sorted)
    ax.set_ylabel(compare_metric)
    ax.set_ylim([0, 1.1])
    ax.set_title(f"Cross-Atlas {compare_metric} Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(cfg.plot.save_dir, exist_ok=True)
    for fmt in cfg.plot.formats:
        fig.savefig(
            os.path.join(cfg.plot.save_dir, f"cross_atlas_{compare_metric}.{fmt}"),
            dpi=cfg.plot.dpi, bbox_inches="tight",
        )
    plt.show()


def build_latex_table(
    all_results: Dict[str, Any],
    atlas: str,
    metrics: Optional[List[str]] = None,
) -> str:
    if metrics is None:
        metrics = ["Accuracy", "Sensitivity", "Specificity", "F1", "AUROC", "AUPRC"]

    models_in_atlas = [
        res["model"] for key, res in all_results.items()
        if res["atlas"] == atlas
    ]
    models_in_atlas = [m for m in ALL_MODELS if m in models_in_atlas]

    col_spec = "l" + "c" * len(metrics)
    header = " & ".join(["Model"] + metrics) + r" \\"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        rf"\caption{{ASD Classification Results — {atlas} Atlas (10-Fold CV, Pooled OOF)}}",
        rf"\label{{tab:results_{atlas.lower()}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for model_name in models_in_atlas:
        key = f"{model_name}_{atlas}"
        if key not in all_results:
            continue
        ci = all_results[key]["ci"]
        vals = []
        for m in metrics:
            if m in ci:
                mean_, lo_, hi_ = ci[m]
                vals.append(rf"{mean_:.3f} [{lo_:.3f}--{hi_:.3f}]")
            else:
                vals.append("—")
        lines.append(model_name + " & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)
