# abide_gnnbench/utils/data.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from abide_gnnbench.config import DataConfig


def load_abide(path: str, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    df["SUBJECT_IDX"] = np.arange(len(df))

    if df[cfg.label_col].max() == 2:
        df["LABEL"] = (df[cfg.label_col] == 1).astype(int)
    else:
        df["LABEL"] = df[cfg.label_col].astype(int)

    non_feature_cols = cfg.phenotypic_cols + ["LABEL"]
    fc_cols = [
        c for c in df.columns
        if c.startswith("FC_")
    ]

    print(f"Total subjects       : {len(df)}")
    print(f"FC features          : {len(fc_cols)}")
    print(f"Class distribution:\n{df['LABEL'].value_counts().rename({1: 'ASD', 0: 'TD'})}")
    print(f"Missing FC values    : {df[fc_cols].isnull().sum().sum()}")
    print(f"FC |x| > 4 count     : {(df[fc_cols].abs() > 4).sum().sum()}")

    X = df[fc_cols].values.astype(np.float32)
    y = df["LABEL"].values.astype(np.float32)
    site_ids = df[cfg.site_col].values
    subject_idx = df["SUBJECT_IDX"].values

    return X, y, site_ids, subject_idx, fc_cols


def scale_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, scaler


def make_val_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(sss.split(X_train, y_train))
    return X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx]
