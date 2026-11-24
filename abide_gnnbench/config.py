# abide_gnnbench/config.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    aal_path: str = "data/ABIDE_AAL_Merged_FC_Phenotypic.csv"
    cc200_path: str = "data/ABIDE_CC200_Merged_FC_Phenotypic.csv"
    phenotypic_cols: List[str] = field(default_factory=lambda: [
        "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ", "SUBJECT_IDX"
    ])
    label_col: str = "DX_GROUP"
    site_col: str = "SITE_ID"


@dataclass
class CVConfig:
    n_splits: int = 10
    tune_folds: int = 10
    val_frac: float = 0.10
    seed: int = 42


@dataclass
class TrainConfig:
    max_epochs: int = 100
    patience: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class BootstrapConfig:
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 42


@dataclass
class GraphConfig:
    # Threshold for FC matrix binarization / edge construction
    threshold_method: str = "percentile"   # "percentile" | "absolute"
    threshold_value: float = 90.0          # top-10% edges if percentile
    self_loops: bool = True
    edge_weight: bool = True


@dataclass
class ExplainConfig:
    shap_bg_size: int = 100
    shap_explain_size: int = 50
    perm_n_repeats: int = 5
    ig_n_steps: int = 50
    top_k_features: int = 20
    gnn_explain_epochs: int = 200


@dataclass
class PlotConfig:
    dpi: int = 150
    figsize_wide: tuple = (14, 11)
    figsize_medium: tuple = (14, 6)
    figsize_square: tuple = (10, 7)
    style: str = "seaborn-v0_8-whitegrid"
    save_dir: str = "results/plots"
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class OutputConfig:
    base_dir: str = "results"
    models_dir: str = "results/models"
    metrics_dir: str = "results/metrics"
    plots_dir: str = "results/plots"
    explainability_dir: str = "results/explainability"
    tuning_dir: str = "results/tuning"


@dataclass
class BenchmarkConfig:
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "auto"   # "auto" | "cpu" | "cuda"
    seed: int = 42


# ── Model-specific default hyperparameters ────────────────────────────────────

MLP_DEFAULT_CFG: Dict[str, Any] = dict(
    hidden_dims=[256, 128],
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=200,
    patience=200,
    use_batchnorm=True,
)

GCN_DEFAULT_CFG: Dict[str, Any] = dict(
    hidden_dims=[256, 128],
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=200,
    patience=200,
)

GAT_DEFAULT_CFG: Dict[str, Any] = dict(
    hidden_dims=[128, 64],
    num_layers=2,
    heads=8,
    dropout=0.4,
    attn_dropout=0.4,
    lr=0.000871768952883686,
    weight_decay=3.96886537719217e-06,
    batch_size=64,
    max_epochs=150,
    patience=150,
    residual=True,
    normalization="graph",  # "none" | "batch" | "layer" | "graph"
)

CHEBNET_DEFAULT_CFG: Dict[str, Any] = dict(
    hidden_dims=[256, 128],
    K=3,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=200,
    patience=200,
)

GIN_DEFAULT_CFG: Dict[str, Any] = dict(
    hidden_dims=[256, 128],
    dropout=0.3,
    eps=0.0,
    train_eps=True,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=200,
    patience=200,
)

GRAPHSAGE_DEFAULT_CFG: Dict[str, Any] = dict(
    hidden_dims=[256, 128],
    dropout=0.3,
    aggr="mean",
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=200,
    patience=200,
)

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "MLP": MLP_DEFAULT_CFG,
    "GCN": GCN_DEFAULT_CFG,
    "GAT": GAT_DEFAULT_CFG,
    "ChebNet": CHEBNET_DEFAULT_CFG,
    "GIN": GIN_DEFAULT_CFG,
    "GraphSAGE": GRAPHSAGE_DEFAULT_CFG,
}

# ── Optuna search spaces ──────────────────────────────────────────────────────

MLP_SEARCH_SPACE: Dict[str, Any] = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128], [512, 256], [256, 128, 64]],
    },
    "dropout": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.1},
    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "use_batchnorm": {"type": "categorical", "choices": [True, False]},
}

GCN_SEARCH_SPACE: Dict[str, Any] = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128], [512, 256]],
    },
    "dropout": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.1},
    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
}

GAT_SEARCH_SPACE: Dict[str, Any] = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128], [512, 256]],
    },
    "heads": {"type": "categorical", "choices": [2, 4, 8]},
    "dropout": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.1},
    "attn_dropout": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
}

CHEBNET_SEARCH_SPACE: Dict[str, Any] = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128], [512, 256]],
    },
    "K": {"type": "int", "low": 2, "high": 6},
    "dropout": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.1},
    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
}

GIN_SEARCH_SPACE: Dict[str, Any] = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128], [512, 256]],
    },
    "dropout": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.1},
    "eps": {"type": "float", "low": 0.0, "high": 0.5},
    "train_eps": {"type": "categorical", "choices": [True, False]},
    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
}

GRAPHSAGE_SEARCH_SPACE: Dict[str, Any] = {
    "hidden_dims": {
        "type": "categorical",
        "choices": [[128, 64], [256, 128], [512, 256]],
    },
    "dropout": {"type": "float", "low": 0.1, "high": 0.6, "step": 0.1},
    "aggr": {"type": "categorical", "choices": ["mean", "max"]},
    "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
}

MODEL_SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "MLP": MLP_SEARCH_SPACE,
    "GCN": GCN_SEARCH_SPACE,
    "GAT": GAT_SEARCH_SPACE,
    "ChebNet": CHEBNET_SEARCH_SPACE,
    "GIN": GIN_SEARCH_SPACE,
    "GraphSAGE": GRAPHSAGE_SEARCH_SPACE,
}
