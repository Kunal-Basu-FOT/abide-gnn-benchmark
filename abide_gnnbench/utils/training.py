# abide_gnnbench/utils/training.py

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.data import Data
    _HAS_PYGEOM = True
except ImportError:
    _HAS_PYGEOM = False


History = Dict[str, list]


# ── MLP / tabular training ─────────────────────────────────────────────────────

def make_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_tabular(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 150,
    patience: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, History]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    safe_batch = min(batch_size, len(X_tr) - 1)  # ensure at least 1 sample left after drop
    loader = make_tensor_loader(X_tr, y_tr, batch_size=safe_batch, shuffle=True, drop_last=True)

    history: History = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_auroc": [],
    }
    best_val_loss = np.inf
    best_weights = None
    no_improve = 0

    for _ in range(max_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            tr_correct += (preds == yb).sum().item()
            tr_total += len(xb)

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val).to(device)
            yv = torch.tensor(y_val).to(device)
            logits_v = model(xv)
            val_loss = criterion(logits_v, yv).item()
            probs_v = torch.sigmoid(logits_v).cpu().numpy()
            preds_v = (probs_v >= 0.5).astype(float)
            val_acc = accuracy_score(y_val, preds_v)
            try:
                val_auroc = roc_auc_score(y_val, probs_v)
            except Exception:
                val_auroc = 0.5

        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_correct / tr_total)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    return model, history


def predict_tabular(
    model: nn.Module,
    X_test: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test).to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    probs = np.atleast_1d(probs)          # <-- FIX: guard against 0-d output (batch size 1)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


# ── GNN / graph training ───────────────────────────────────────────────────────

def train_graph(
    model: nn.Module,
    train_graphs,
    val_graphs,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 150,
    patience: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, History]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)

    history: History = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_auroc": [],
    }
    best_val_loss = np.inf
    best_weights = None
    no_improve = 0

    for _ in range(max_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch.num_graphs
            preds = (torch.sigmoid(logits) >= 0.5).float()
            tr_correct += (preds == batch.y).sum().item()
            tr_total += batch.num_graphs

        model.eval()
        with torch.no_grad():
            all_probs, all_labels = [], []
            val_loss_total = 0.0
            for batch in val_loader:
                batch = batch.to(device)
                logits_v = model(batch)
                val_loss_total += criterion(logits_v, batch.y).item() * batch.num_graphs
                probs_v = torch.sigmoid(logits_v).cpu().numpy()
                all_probs.extend(probs_v.tolist())
                all_labels.extend(batch.y.cpu().numpy().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        val_loss = val_loss_total / len(val_graphs)
        preds_v = (all_probs >= 0.5).astype(float)
        val_acc = accuracy_score(all_labels, preds_v)
        try:
            val_auroc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auroc = 0.5

        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_correct / tr_total)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    return model, history


def predict_graph(
    model: nn.Module,
    graphs,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = GeoDataLoader(graphs, batch_size=len(graphs), shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs = np.atleast_1d(probs)   # <-- FIX: guard against 0-d output
            all_probs.extend(probs.tolist())
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    return all_probs, preds
