"""
Lab 06 — Classification metrics that survive imbalance (Chapter 6: Classification & Its Metrics)
================================================================================================
Chapter: 06 — Classification & Its Metrics

THE PROBLEM
-----------
PayLane, a payments start-up, loses about €200 on every fraudulent transaction that
slips through and spends about €5 of analyst time on every legitimate transaction it
blocks for review. Fraud is 5% of traffic. The previous data scientist reported "95.2%
accuracy" and was congratulated, until someone noticed that blocking nothing at all
scores 95.0%. You are asked to rebuild the evaluation from the four counts up: the
confusion matrix, precision and recall written by hand and checked against
scikit-learn, ROC-AUC computed as the ranking statistic it really is, and a threshold
chosen by expected value under PayLane's cost matrix rather than the default 0.5.
The same toolkit is then applied to a clinic's tumour screen, where the business rule
is different: missing a malignant tumour is unacceptable, so recall must be at least
98% and the threshold that guarantees it is tuned on out-of-fold probabilities, never
on the test set.

TASKS (make the tests in labs/tests/test_ch06.py pass, one at a time)
---------------------------------------------------------------------
1. sigmoid(z)                                       → score -> probability, numerically safe
2. log_loss(y, p)                                   → mean cross-entropy with clipping
3. confusion_counts(y_true, y_pred)                 → {"tn", "fp", "fn", "tp"} from scratch
4. classification_metrics(y_true, y_pred)           → accuracy, precision, recall, specificity, f1
5. roc_auc(y_true, scores)                          → AUC as the fraction of correctly ranked pairs
6. best_threshold_by_value(y_true, proba, values)   → threshold maximising expected value
7. tumour_screen(seed, min_recall)                  → full sklearn model + recall-first threshold

STRETCH (no tests)
------------------
- Compute average precision from the same rank statistic idea and compare with
  sklearn.metrics.average_precision_score on the fraud data.
- Double the fraud cost to €400 and report how far the optimal threshold moves; then halve the
  review cost. Which estimate does the decision depend on more?

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np

# PayLane's value matrix (euros per transaction; costs are negative)
PAYLANE_VALUES = {"tp": 0.0, "tn": 0.0, "fp": -5.0, "fn": -200.0}


def make_fraud_scores(seed: int = 7, n: int = 4000, fraud_rate: float = 0.05):
    """Held-out labels and model probabilities for n transactions (5% fraud). Do not modify.

    Simulates a decent-but-imperfect model: fraud scores centre higher than legit scores,
    with overlap. Returns (y_true, proba).
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    z = np.where(y == 1, rng.normal(0.8, 1.3, n), rng.normal(-2.4, 1.1, n))
    return y, 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- Task 1
def sigmoid(z) -> np.ndarray:
    """σ(z) = 1 / (1 + exp(-z)), vectorised (chapter §2).

    Must not overflow for z = -1000 (np.exp(1000) is inf): clip z to [-500, 500] first, or
    use the two-branch form. Returns an ndarray (float) even for a scalar input.

    Example: sigmoid([-4, 0, 4]) -> [0.018, 0.5, 0.982]
    """
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- Task 2
def log_loss(y, p, eps: float = 1e-15) -> float:
    """Mean cross-entropy: -mean(y log p + (1-y) log(1-p))  (chapter §2 and §10).

    Clip p to [eps, 1-eps] first so a confident wrong answer costs a lot but not inf.
    Matches sklearn.metrics.log_loss for binary labels.

    Example: y = [1, 0], p = [0.9, 0.1] -> 0.105 ; y = [0], p = [0.999] -> 6.91
    """
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- Task 3
def confusion_counts(y_true, y_pred) -> dict:
    """The four counts (chapter §4). Positive class is 1.

    Returns {"tn": ..., "fp": ..., "fn": ..., "tp": ...} as Python ints; the same numbers as
    confusion_matrix(y_true, y_pred).ravel() in sklearn's [[TN, FP], [FN, TP]] layout.

    Example: y_true = [1, 1, 0, 0], y_pred = [1, 0, 1, 0] -> tp 1, fn 1, fp 1, tn 1
    """
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {"tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "tp": int(((y_true == 1) & (y_pred == 1)).sum())}


# ---------------------------------------------------------------- Task 4
def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall, specificity and F1 from the four counts (chapter §5).

    precision = tp / (tp + fp); recall = tp / (tp + fn); specificity = tn / (tn + fp);
    f1 = 2 * precision * recall / (precision + recall). Any 0/0 is 0.0 (sklearn's zero_division=0).
    Returns {"accuracy", "precision", "recall", "specificity", "f1"} as floats.

    Example: the chapter's 1,000 transactions (tp 35, fn 15, fp 20, tn 930) ->
             accuracy 0.965, precision 0.636, recall 0.70, specificity 0.979, f1 0.667
    """
    c = confusion_counts(y_true, y_pred)
    n = sum(c.values())

    def safe(num, den):
        return float(num / den) if den > 0 else 0.0

    precision = safe(c["tp"], c["tp"] + c["fp"])
    recall = safe(c["tp"], c["tp"] + c["fn"])
    return {"accuracy": safe(c["tp"] + c["tn"], n), "precision": precision, "recall": recall,
            "specificity": safe(c["tn"], c["tn"] + c["fp"]),
            "f1": safe(2 * precision * recall, precision + recall)}


# ---------------------------------------------------------------- Task 5
def roc_auc(y_true, scores) -> float:
    """AUC as a rank statistic (chapter §7): P(random positive scores above random negative).

    Count every (positive, negative) pair: +1 if the positive's score is higher, +0.5 on a
    tie, 0 otherwise; divide by n_pos * n_neg. Vectorise with broadcasting
    (scores[pos][:, None] vs scores[neg][None, :]) or use ranks. Must match
    sklearn.metrics.roc_auc_score, ties included.

    Example: chapter exercise 2, scores descending D, D, R, D, R, R, R, R -> 14/15 = 0.933
    """
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


# ---------------------------------------------------------------- Task 6
def best_threshold_by_value(y_true, proba, values: dict = PAYLANE_VALUES,
                            grid=None) -> tuple[float, float]:
    """Sweep thresholds and return the one with the highest total expected value (chapter §8).

    value(t) = tp*values["tp"] + tn*values["tn"] + fp*values["fp"] + fn*values["fn"], with the
    counts from predicting 1 where proba >= t. grid defaults to np.arange(0.01, 1.0, 0.01).
    Returns (best_threshold, best_value); ties keep the first (lowest) threshold.
    When a miss costs 40x a false alarm the optimum sits far below 0.5, and it is NOT the
    threshold that maximises accuracy.

    Example: y = [1, 0], proba = [0.3, 0.2], values fp=-5, fn=-200 -> threshold 0.21..0.3, value 0.0
    """
    y_true, proba = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    grid = np.arange(0.01, 1.0, 0.01) if grid is None else np.asarray(grid, dtype=float)
    best_t, best_v = None, -np.inf
    for t in grid:
        c = confusion_counts(y_true, (proba >= t).astype(int))
        v = sum(c[k] * values[k] for k in ("tp", "tn", "fp", "fn"))
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


# ---------------------------------------------------------------- Task 7
def tumour_screen(seed: int = 42, min_recall: float = 0.98) -> dict:
    """The chapter's breast-cancer model with a recall-first threshold (§2, §7, §8).

    Steps: load_breast_cancer; y = 1 - y so that 1 = malignant; train_test_split(test_size=0.3,
    stratify=y, random_state=seed); model = make_pipeline(StandardScaler(),
    LogisticRegression(max_iter=1000)).
    Threshold tuning must not touch the test set: get out-of-fold probabilities on the
    TRAINING rows with cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1],
    then choose the HIGHEST threshold in np.arange(0.01, 0.51, 0.01) whose recall on those
    out-of-fold probabilities is >= min_recall (fall back to 0.01 if none). Fit the model on all
    training rows and score the test set at that threshold.
    Returns {"model": fitted pipeline, "threshold": float, "auc": float (test, via roc_auc),
             "metrics": classification_metrics at the threshold on test,
             "metrics_at_0_5": classification_metrics at 0.5 on test,
             "majority_baseline": accuracy of always predicting the majority class on test}.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    threshold = 0.01
    for t in np.arange(0.01, 0.51, 0.01):
        if classification_metrics(y_tr, (oof >= t).astype(int))["recall"] >= min_recall:
            threshold = float(t)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    return {"model": model, "threshold": threshold, "auc": roc_auc(y_te, proba),
            "metrics": classification_metrics(y_te, (proba >= threshold).astype(int)),
            "metrics_at_0_5": classification_metrics(y_te, (proba >= 0.5).astype(int)),
            "majority_baseline": float(max(y_te.mean(), 1 - y_te.mean()))}


if __name__ == "__main__":
    y, p = make_fraud_scores()
    print(f"fraud rate {y.mean():.1%}; log-loss of the model {log_loss(y, p):.3f} "
          f"vs base-rate guess {log_loss(y, np.full_like(p, y.mean())):.3f}")
    print("counts at 0.5:", confusion_counts(y, (p >= 0.5).astype(int)))
    m = classification_metrics(y, (p >= 0.5).astype(int))
    print("metrics at 0.5:", {k: round(v, 3) for k, v in m.items()})
    print("'block nothing' accuracy:", round(classification_metrics(y, np.zeros_like(y))["accuracy"], 3))
    print(f"ROC-AUC (rank statistic): {roc_auc(y, p):.4f}")
    t, v = best_threshold_by_value(y, p)
    _, v_default = best_threshold_by_value(y, p, grid=[0.5])
    print(f"best threshold by expected value: {t:.2f} -> €{v:,.0f} on {len(y)} transactions "
          f"(vs €{v_default:,.0f} at 0.5)")
    res = tumour_screen()
    print(f"\ntumour screen: AUC {res['auc']:.4f}; threshold {res['threshold']:.2f}; "
          f"recall {res['metrics']['recall']:.3f}, precision {res['metrics']['precision']:.3f} "
          f"(at 0.5: recall {res['metrics_at_0_5']['recall']:.3f}); majority baseline {res['majority_baseline']:.3f}")
