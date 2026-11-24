"""Run the ABIDE benchmark using explicit input and output locations."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aal-csv", type=Path)
    parser.add_argument("--cc200-csv", type=Path)
    parser.add_argument("--atlases", default="AAL,CC200")
    parser.add_argument("--models", default="MLP,GCN,ChebNet,GIN,GraphSAGE,GAT")
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    atlases = [name.strip().upper() for name in args.atlases.split(",") if name.strip()]
    if not atlases or any(name not in ("AAL", "CC200") for name in atlases):
        parser.error("--atlases must contain AAL and/or CC200")
    for name in atlases:
        path = args.aal_csv if name == "AAL" else args.cc200_csv
        if path is None or not path.is_file():
            parser.error(f"--{name.lower()}-csv must be an existing file")

    from abide_gnnbench.benchmark import ALL_MODELS, run_full_benchmark
    from abide_gnnbench.config import BenchmarkConfig

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not models or any(name not in ALL_MODELS for name in models):
        parser.error(f"--models must contain names from {', '.join(ALL_MODELS)}")

    cfg = BenchmarkConfig()
    if args.aal_csv:
        cfg.data.aal_path = str(args.aal_csv.resolve())
    if args.cc200_csv:
        cfg.data.cc200_path = str(args.cc200_csv.resolve())
    cfg.device = args.device
    out = args.output_dir.resolve()
    cfg.output.base_dir = str(out)
    cfg.output.metrics_dir = str(out / "metrics")
    cfg.output.models_dir = str(out / "models")
    cfg.output.plots_dir = str(out / "plots")
    cfg.output.explainability_dir = str(out / "explainability")
    cfg.output.tuning_dir = str(out / "tuning")
    cfg.plot.save_dir = cfg.output.plots_dir
    run_full_benchmark(cfg, models, atlases, args.tune, args.explain)


if __name__ == "__main__":
    main()
