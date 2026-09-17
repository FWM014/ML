"""
Lab 01 — Your first models (Chapter 1: What Machine Learning Really Is)
======================================================================

THE PROBLEM
-----------
A grocery chain wants to sort fruit on a conveyor belt using a single scale. You get
weights (grams) and the true fruit type for a labeled batch. Before anyone buys a
camera system, management wants to know: how far can a *one-number* model go, and
how do we measure that honestly?

Then you'll build a real scikit-learn model on a medical dataset and prove it beats a
baseline on data it has never seen — the exact skeleton you'll reuse on every project.

TASKS (make the tests in labs/tests/test_ch01.py pass, one at a time)
---------------------------------------------------------------------
1. baseline_accuracy(y)                → accuracy of always predicting the majority class
2. threshold_predict(x, thr)           → 1 if weight > thr else 0, vectorized
3. best_threshold(x, y, candidates)    → the candidate threshold with the highest accuracy
4. holdout_split(X, y, test_frac, seed)→ shuffle and split without leaking rows
5. beat_the_baseline()                 → DummyClassifier vs LogisticRegression on unseen data

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np


def make_fruit_data(seed: int = 0, n_per_class: int = 60):
    """Weights of apples (label 0) and oranges (label 1). Do not modify."""
    rng = np.random.default_rng(seed)
    apples = rng.normal(155, 22, size=n_per_class)
    oranges = rng.normal(210, 25, size=n_per_class)
    x = np.concatenate([apples, oranges]).round(1)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


# ---------------------------------------------------------------- Task 1
def baseline_accuracy(y: np.ndarray) -> float:
    """Accuracy of a model that always predicts the most common label in y.

    Example: y = [0, 0, 0, 1]  ->  0.75
    """
    y = np.asarray(y)
    counts = np.bincount(y)
    return counts.max() / len(y)


# ---------------------------------------------------------------- Task 2
def threshold_predict(x: np.ndarray, thr: float) -> np.ndarray:
    """Return an int array: 1 where x > thr ("orange"), else 0 ("apple")."""
    return (np.asarray(x) > thr).astype(int)


# ---------------------------------------------------------------- Task 3
def best_threshold(x: np.ndarray, y: np.ndarray, candidates) -> tuple[float, float]:
    """Try every candidate threshold; return (best_threshold, best_accuracy).

    Ties: keep the FIRST candidate that reaches the best accuracy.
    """
    best_thr, best_acc = None, -1.0
    for thr in candidates:
        acc = float((threshold_predict(x, thr) == y).mean())
        if acc > best_acc:
            best_thr, best_acc = float(thr), acc
    return best_thr, best_acc


# ---------------------------------------------------------------- Task 4
def holdout_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.25, seed: int = 0):
    """Shuffle the rows with np.random.default_rng(seed).permutation and split.

    Returns X_train, X_test, y_train, y_test. The test set holds round(test_frac * n) rows.
    Every row appears in exactly one of the two sets.
    """
    X, y = np.asarray(X), np.asarray(y)
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_frac * n))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------- Task 5
def beat_the_baseline(seed: int = 0) -> dict:
    """Train on the breast-cancer dataset and score BOTH models on unseen rows.

    Steps: load_breast_cancer(return_X_y=True) -> train_test_split(test_size=0.25,
    random_state=seed, stratify=y) -> DummyClassifier(strategy="most_frequent") and
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).
    Return {"baseline": <test accuracy>, "model": <test accuracy>}.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    return {"baseline": float(baseline.score(X_te, y_te)), "model": float(model.score(X_te, y_te))}


if __name__ == "__main__":
    x, y = make_fruit_data()
    print(f"baseline accuracy (always 'apple' or always 'orange'): {baseline_accuracy(y):.2%}")
    thr, acc = best_threshold(x, y, candidates=range(100, 300))
    print(f"best single threshold: {thr} g -> accuracy {acc:.2%}")
    X_tr, X_te, y_tr, y_te = holdout_split(x.reshape(-1, 1), y, 0.25, seed=1)
    thr_tr, _ = best_threshold(X_tr.ravel(), y_tr, candidates=range(100, 300))
    print(f"threshold chosen on train ({thr_tr} g) scores {(threshold_predict(X_te.ravel(), thr_tr) == y_te).mean():.2%} on held-out rows")
    print("breast cancer:", beat_the_baseline())
