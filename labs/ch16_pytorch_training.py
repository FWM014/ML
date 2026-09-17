"""
Lab 16 — A training loop you can trust (Chapter 16: Training Deep Networks (PyTorch))
====================================================================================

THE PROBLEM
-----------
Northwind Payments approves ~40,000 card transactions a day. The risk team has a
gradient-boosted tree in production and wants an MLP as a second opinion, but their
last attempt was a mess: the loss curve wobbled, nobody knew whether SGD or Adam was
"better", the model was saved as a pickled object that no longer loads, and the
"92 % accuracy" turned out to be measured on the training set. Your job is to build
the training skeleton the team will reuse on every future model: a configurable MLP,
a loop with validation + early stopping that restores the best weights, an optimizer
bake-off on identical starting weights, a warmup + cosine LR schedule, a
regularization experiment that provably shrinks the train/validation gap, and a
state_dict save/load round-trip. The data is a seeded synthetic stand-in for the real
feature table (≤ 500 rows); every experiment must run on a laptop CPU in seconds.

TASKS (make the tests in labs/tests/test_ch16.py pass, one at a time)
---------------------------------------------------------------------
1. build_mlp(d_in, hidden, d_out, dropout)     → nn.Module of Linear/ReLU/Dropout ending in LOGITS
2. train_with_early_stopping(...)              → history dict; best state restored; patience
3. compare_optimizers(X, y, steps, seed)       → {"sgd": final_loss, "adam": final_loss} from identical init
4. lr_schedule(base_lr, warmup, total)         → list of per-epoch LRs from a LambdaLR (warmup + cosine)
5. regularization_gap(seed)                    → dropout + weight decay shrink the train/val loss gap
6. save_and_reload(model, builder, path)       → state_dict round-trip; the reloaded model is identical

STRETCH (no tests)
------------------
- Add nn.BatchNorm1d after each hidden Linear in build_mlp and re-run task 5. Does the
  gap shrink further, and what happens if you forget model.eval() before validation?
- Run the LR range test from §5 on make_fraud_data and pick lr for task 2 from the curve.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tiny models: one thread beats thread-sync overhead and keeps timings reproducible.
torch.set_num_threads(1)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _to_tensor(a, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=dtype)


def make_fraud_data(seed: int = 0, n: int = 500):
    """A stand-in for the transaction table: two curved classes, scaled on TRAIN only.

    Returns X_tr, X_va, y_tr, y_va as float32 tensors; y has shape (n, 1). Do not modify.
    """
    X, y = make_moons(n_samples=n, noise=0.25, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    sc = StandardScaler().fit(X_tr)
    return (_to_tensor(sc.transform(X_tr)), _to_tensor(sc.transform(X_va)),
            _to_tensor(y_tr).unsqueeze(1), _to_tensor(y_va).unsqueeze(1))


def make_overfit_data(seed: int = 0, n: int = 200, d: int = 40):
    """A tiny, wide table: 40 features, only 5 informative, 100 training rows.

    A 64x64 MLP memorizes it in a few epochs -> big train/val gap. Do not modify.
    """
    X, y = make_classification(n_samples=n, n_features=d, n_informative=5, n_redundant=0,
                               n_clusters_per_class=2, flip_y=0.05, class_sep=1.2, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    sc = StandardScaler().fit(X_tr)
    return (_to_tensor(sc.transform(X_tr)), _to_tensor(sc.transform(X_va)),
            _to_tensor(y_tr).unsqueeze(1), _to_tensor(y_va).unsqueeze(1))


# ---------------------------------------------------------------- Task 1
def build_mlp(d_in: int, hidden: tuple[int, ...] = (32, 32), d_out: int = 1,
              dropout: float = 0.0) -> nn.Module:
    """Return an nn.Sequential MLP: [Linear -> ReLU (-> Dropout if dropout > 0)] per hidden width,
    then one final Linear(last_hidden, d_out). NO sigmoid/softmax at the end: the loss
    function (BCEWithLogitsLoss / CrossEntropyLoss) owns it (§3).

    Example: build_mlp(2, (32, 16), 1) has 2*32+32 + 32*16+16 + 16*1+1 = 657 parameters
    and maps a (n, 2) batch to (n, 1) logits.
    """
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- Task 2
def train_with_early_stopping(model: nn.Module, X_tr: torch.Tensor, y_tr: torch.Tensor,
                              X_va: torch.Tensor, y_va: torch.Tensor, lr: float = 1e-2,
                              max_epochs: int = 30, patience: int = 5, batch_size: int = 32,
                              weight_decay: float = 0.0, seed: int = 0) -> dict:
    """The §3 loop: mini-batch Adam (or AdamW when weight_decay > 0) with BCEWithLogitsLoss,
    then validation in eval()/no_grad() after every epoch, and early stopping.

    - torch.manual_seed(seed) first so the batch order is reproducible.
    - Per epoch: model.train(); shuffle with torch.randperm; for each batch:
      forward -> loss -> zero_grad -> backward -> step.
    - After each epoch: model.eval(); record the full-batch train loss and val loss.
    - Keep a copy of state_dict() whenever val loss improves by more than 1e-4; stop
      when it has not improved for `patience` epochs; load the best state back.
    - Leave the model in eval() mode.

    Returns {"train_loss": [...], "val_loss": [...], "best_epoch": int (1-based),
             "epochs_run": int}. len(train_loss) == len(val_loss) == epochs_run.
    """
    torch.manual_seed(seed)
    loss_fn = nn.BCEWithLogitsLoss()
    if weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "epochs_run": 0}
    best_val, best_state, bad = float("inf"), None, 0
    n = len(X_tr)
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = loss_fn(model(X_tr), y_tr).item()
            va = loss_fn(model(X_va), y_va).item()
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["epochs_run"] = epoch
        if va < best_val - 1e-4:
            best_val, bad = va, 0
            history["best_epoch"] = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history


# ---------------------------------------------------------------- Task 3
def compare_optimizers(X: torch.Tensor, y: torch.Tensor, steps: int = 60, seed: int = 0) -> dict:
    """§4: same net, same starting weights, same data — SGD(lr=0.05) vs Adam(lr=0.01).

    For each optimizer: torch.manual_seed(seed) *immediately before* build_mlp(d_in, (16,), 1)
    so both start from identical weights; then `steps` full-batch steps of
    BCEWithLogitsLoss; return the final loss (a float).

    Returns {"sgd": float, "adam": float}. Calling twice with the same seed must give the
    same numbers.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    out = {}
    for name, cls, kw in [("sgd", torch.optim.SGD, dict(lr=0.05)), ("adam", torch.optim.Adam, dict(lr=0.01))]:
        torch.manual_seed(seed)
        net = build_mlp(X.shape[1], (16,), 1)
        opt = cls(net.parameters(), **kw)
        for _ in range(steps):
            loss = loss_fn(net(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        out[name] = float(loss.item())
    return out


# ---------------------------------------------------------------- Task 4
def lr_schedule(base_lr: float = 0.1, warmup: int = 3, total: int = 12) -> list[float]:
    """§5: warmup + cosine decay, driven by torch.optim.lr_scheduler.LambdaLR.

    The multiplier for epoch e (0-based) is
        (e + 1) / warmup                                        if e < warmup
        0.5 * (1 + cos(pi * (e - warmup) / (total - warmup)))   otherwise
    Build any tiny optimizer (e.g. SGD on nn.Linear(4, 1) with lr=base_lr), wrap it in
    LambdaLR, and return the LR *in effect during* each epoch 0..total-1: read
    opt.param_groups[0]["lr"] at the start of the epoch, then opt.step(); sched.step().

    Example: lr_schedule(0.1, 3, 12)[:4] == [0.0333, 0.0667, 0.1, 0.1] (rounded), and the
    last value is ~0.003 — the schedule never reaches exactly 0 inside the loop.
    """
    net = nn.Linear(4, 1)
    opt = torch.optim.SGD(net.parameters(), lr=base_lr)

    def lr_lambda(e: int) -> float:
        if e < warmup:
            return (e + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (e - warmup) / (total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lrs = []
    for _ in range(total):
        lrs.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        sched.step()
    return lrs


# ---------------------------------------------------------------- Task 5
def regularization_gap(seed: int = 0, epochs: int = 30) -> dict:
    """§7: on make_overfit_data(seed), train two MLPs of the same size, hidden=(64, 64):

    - "plain":       dropout=0.0, weight_decay=0.0
    - "regularized": dropout=0.5, weight_decay=0.3   (AdamW: decoupled weight decay)
    both with train_with_early_stopping(lr=1e-3, max_epochs=epochs, patience=epochs
    (i.e. never stop early), batch_size=16, seed=seed), seeding torch.manual_seed(seed)
    before each build_mlp.

    For each, report the LAST epoch's train and val loss and their gap:
    {"plain": {"train_loss", "val_loss", "gap"}, "regularized": {...}} with
    gap = val_loss - train_loss. The regularized gap must be clearly smaller.
    """
    X_tr, X_va, y_tr, y_va = make_overfit_data(seed)
    result = {}
    for name, p_drop, wd in [("plain", 0.0, 0.0), ("regularized", 0.5, 0.3)]:
        torch.manual_seed(seed)
        model = build_mlp(X_tr.shape[1], (64, 64), 1, dropout=p_drop)
        hist = train_with_early_stopping(model, X_tr, y_tr, X_va, y_va, lr=1e-3, max_epochs=epochs,
                                         patience=epochs, batch_size=16, weight_decay=wd, seed=seed)
        tr, va = hist["train_loss"][-1], hist["val_loss"][-1]
        result[name] = {"train_loss": tr, "val_loss": va, "gap": va - tr}
    return result


# ---------------------------------------------------------------- Task 6
def save_and_reload(model: nn.Module, builder, path: str) -> nn.Module:
    """§9: persist the state_dict (never the pickled object), rebuild the architecture with
    builder() (a zero-argument function), load the state, return the fresh model in eval().

    - os.makedirs(os.path.dirname(path), exist_ok=True) so labs/data/ exists.
    - torch.save(model.state_dict(), path); fresh = builder();
      fresh.load_state_dict(torch.load(path)); fresh.eval().
    The reloaded model must give torch.allclose outputs to the original on any input.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    fresh = builder()
    fresh.load_state_dict(torch.load(path))
    fresh.eval()
    return fresh


if __name__ == "__main__":
    X_tr, X_va, y_tr, y_va = make_fraud_data()
    torch.manual_seed(0)
    model = build_mlp(2, (32, 32), 1)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    hist = train_with_early_stopping(model, X_tr, y_tr, X_va, y_va, lr=1e-2)
    print(f"early stopping: ran {hist['epochs_run']} epochs, best epoch {hist['best_epoch']}, "
          f"val loss {min(hist['val_loss']):.3f} (first epoch {hist['val_loss'][0]:.3f})")
    with torch.no_grad():
        acc = ((model(X_va) > 0).float() == y_va).float().mean().item()
    print(f"validation accuracy: {acc:.3f}")
    print("optimizer bake-off (60 full-batch steps):", {k: round(v, 3) for k, v in compare_optimizers(X_tr, y_tr).items()})
    print("LR schedule:", [round(v, 4) for v in lr_schedule(0.1, 3, 12)])
    gaps = regularization_gap()
    for k, v in gaps.items():
        print(f"{k:12s} train {v['train_loss']:.3f}  val {v['val_loss']:.3f}  gap {v['gap']:.3f}")
    fresh = save_and_reload(model, lambda: build_mlp(2, (32, 32), 1), os.path.join(DATA_DIR, "ch16_mlp.pt"))
    with torch.no_grad():
        print("reloaded model identical:", torch.allclose(fresh(X_va), model(X_va)))
