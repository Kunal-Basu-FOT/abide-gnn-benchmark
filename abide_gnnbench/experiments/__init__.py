# abide_gnnbench/experiments/__init__.py

from abide_gnnbench.experiments.run_mlp import run_mlp_experiment
from abide_gnnbench.experiments.run_gnn import run_gnn_experiment

__all__ = ["run_mlp_experiment", "run_gnn_experiment"]
