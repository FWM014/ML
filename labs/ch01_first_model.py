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
    # TODO: count each label (np.bincount) and divide the largest count by len(y)
    raise NotImplementedError("Task: baseline_accuracy")


# ---------------------------------------------------------------- Task 2
def threshold_predict(x: np.ndarray, thr: float) -> np.ndarray:
    """Return an int array: 1 where x > thr ("orange"), else 0 ("apple")."""
    # TODO: compare x > thr and convert the booleans to int
    raise NotImplementedError("Task: threshold_predict")


# ---------------------------------------------------------------- Task 3
def best_threshold(x: np.ndarray, y: np.ndarray, candidates) -> tuple[float, float]:
    """Try every candidate threshold; return (best_threshold, best_accuracy).

    Ties: keep the FIRST candidate that reaches the best accuracy.
    """
    # TODO: loop over candidates, compute accuracy with threshold_predict, keep the best (first on ties)
    raise NotImplementedError("Task: best_threshold")


# ---------------------------------------------------------------- Task 4
def holdout_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.25, seed: int = 0):
    """Shuffle the rows with np.random.default_rng(seed).permutation and split.

    Returns X_train, X_test, y_train, y_test. The test set holds round(test_frac * n) rows.
    Every row appears in exactly one of the two sets.
    """
    # TODO: permute indices with the seeded Generator, take the first round(test_frac*n) as test
    raise NotImplementedError("Task: holdout_split")


# ---------------------------------------------------------------- Task 5
def beat_the_baseline(seed: int = 0) -> dict:
    """Train on the breast-cancer dataset and score BOTH models on unseen rows.

    Steps: load_breast_cancer(return_X_y=True) -> train_test_split(test_size=0.25,
    random_state=seed, stratify=y) -> DummyClassifier(strategy="most_frequent") and
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).
    Return {"baseline": <test accuracy>, "model": <test accuracy>}.
    """
    # TODO: follow the steps in the docstring; return a dict with keys "baseline" and "model"
    raise NotImplementedError("Task: beat_the_baseline")


if __name__ == "__main__":
    x, y = make_fruit_data()
    print(f"baseline accuracy (always 'apple' or always 'orange'): {baseline_accuracy(y):.2%}")
    thr, acc = best_threshold(x, y, candidates=range(100, 300))
    print(f"best single threshold: {thr} g -> accuracy {acc:.2%}")
    X_tr, X_te, y_tr, y_te = holdout_split(x.reshape(-1, 1), y, 0.25, seed=1)
    thr_tr, _ = best_threshold(X_tr.ravel(), y_tr, candidates=range(100, 300))
    print(f"threshold chosen on train ({thr_tr} g) scores {(threshold_predict(X_te.ravel(), thr_tr) == y_te).mean():.2%} on held-out rows")
    print("breast cancer:", beat_the_baseline())
