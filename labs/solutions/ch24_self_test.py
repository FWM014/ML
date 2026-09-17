"""
Lab 24 — The mastery check (Chapter: 24 — Cheat Sheets, Glossary & Interview Prep)
==================================================================================

THE PROBLEM
-----------
You are interviewing for a senior ML role at Halden Analytics. The take-home is not
"train the best model"; it is the set of judgement calls the book has been drilling for
twenty-three chapters, encoded as functions so a reviewer can run them. Given a confusion
matrix, produce every metric by hand. Given a one-paragraph description of a problem,
name the metric. Given a data profile, name the first model to try. Given three numbers
(train, validation, production), name the disease. Given a raw DataFrame, run the leakage
checklist before anyone believes a score. Finally, on a small public dataset, beat a
sensible baseline by a stated margin *without* touching the test set until the end —
and prove it.

TASKS (make the tests in labs/tests/test_ch24.py pass, one at a time)
---------------------------------------------------------------------
1. metrics_from_confusion(tp, fp, fn, tn) → accuracy, precision, recall, specificity, F1, MCC (Section 5)
2. which_metric(problem)                  → the default metric for a problem description (Section 4)
3. which_model_first(profile)             → the first model to try for a data profile (Section 4, Figure 24.3)
4. diagnose(train, val, prod)             → "underfit" / "overfit" / "leakage-or-drift" / "healthy" (Section 8)
5. leakage_checklist(df, target, as_of)   → id-like, constant, target-duplicate and future-dated columns (Section 2)
6. beat_the_baseline_by(margin)           → a leak-free pipeline that beats a stump by the margin on unseen data

STRETCH
-------
- Extend diagnose with a validation standard deviation and a fifth answer, "unstable".
- Add a "groups" check to leakage_checklist: given an entity column, report whether the same entity
  would land in both halves of a random split.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

METRICS = ["MAE", "RMSE", "PR-AUC", "ROC-AUC", "Brier", "macro-F1", "NDCG@k", "MASE", "silhouette", "ARI"]
MODELS = ["logistic/linear", "gradient boosting", "CNN transfer learning", "TF-IDF + linear",
          "fine-tuned transformer", "seasonal naive, then GBM on lag features", "Isolation Forest", "k-means / PCA"]


# ---------------------------------------------------------------- Task 1
def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Every threshold metric from the four cells of a binary confusion matrix.

    accuracy = (tp+tn)/n, precision = tp/(tp+fp), recall = tp/(tp+fn), specificity = tn/(tn+fp),
    f1 = 2PR/(P+R), mcc = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)).
    Any ratio with a zero denominator is 0.0. Return plain floats.
    Example: tp=43, fp=1, fn=9, tn=447 -> precision 0.977, recall 0.827, f1 0.896, mcc 0.889
    """
    # Approach: a small safe_div helper; compute the six numbers; return a dict
    def div(a, b):
        return float(a / b) if b else 0.0

    n = tp + fp + fn + tn
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": div(tp + tn, n), "precision": precision, "recall": recall,
            "specificity": div(tn, tn + fp), "f1": div(2 * precision * recall, precision + recall),
            "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0}


# ---------------------------------------------------------------- Task 2
def which_metric(problem: dict) -> str:
    """The default metric for a problem, following the table in Section 4. Return one of METRICS.

    ``problem`` has "task" in {"regression", "binary", "multiclass", "ranking", "forecasting", "clustering"}
    and optional flags: "outliers" (bool), "positive_rate" (float), "probabilities_matter" (bool),
    "labels_available" (bool). Rules, in this order:
      regression  -> "MAE" if outliers else "RMSE"
      binary      -> "Brier" if probabilities_matter; else "PR-AUC" if positive_rate < 0.2 or > 0.8; else "ROC-AUC"
      multiclass  -> "macro-F1"
      ranking     -> "NDCG@k"
      forecasting -> "MASE"
      clustering  -> "ARI" if labels_available else "silhouette"
    Example: {"task": "binary", "positive_rate": 0.01} -> "PR-AUC"
    """
    # Approach: a chain of if/elif on problem["task"] with .get(...) for the optional flags
    task = problem["task"]
    if task == "regression":
        return "MAE" if problem.get("outliers") else "RMSE"
    if task == "binary":
        if problem.get("probabilities_matter"):
            return "Brier"
        rate = problem.get("positive_rate", 0.5)
        return "PR-AUC" if (rate < 0.2 or rate > 0.8) else "ROC-AUC"
    if task == "multiclass":
        return "macro-F1"
    if task == "ranking":
        return "NDCG@k"
    if task == "forecasting":
        return "MASE"
    if task == "clustering":
        return "ARI" if problem.get("labels_available") else "silhouette"
    raise ValueError(f"unknown task {task!r}")


# ---------------------------------------------------------------- Task 3
def which_model_first(profile: dict) -> str:
    """The first model to try, following Figure 24.3. Return one of MODELS.

    ``profile`` has "input" in {"tabular", "images", "audio", "text", "time_series"} and optional
    "n_rows" (int), "labeled" (bool, default True), "must_explain" (bool), "goal" ("anomaly" or other).
      images / audio            -> "CNN transfer learning"
      text                      -> "TF-IDF + linear" if n_rows < 50_000 else "fine-tuned transformer"
      time_series               -> "seasonal naive, then GBM on lag features"
      tabular, not labeled      -> "Isolation Forest" if goal == "anomaly" else "k-means / PCA"
      tabular, labeled          -> "logistic/linear" if n_rows < 1000 or must_explain else "gradient boosting"
    Example: {"input": "tabular", "n_rows": 50_000, "labeled": True} -> "gradient boosting"
    """
    # Approach: branch on profile["input"]; defaults via .get; keep the strings exactly as in MODELS
    inp = profile["input"]
    n = profile.get("n_rows", 0)
    if inp in ("images", "audio"):
        return "CNN transfer learning"
    if inp == "text":
        return "TF-IDF + linear" if n < 50_000 else "fine-tuned transformer"
    if inp == "time_series":
        return "seasonal naive, then GBM on lag features"
    if inp == "tabular":
        if not profile.get("labeled", True):
            return "Isolation Forest" if profile.get("goal") == "anomaly" else "k-means / PCA"
        return "logistic/linear" if (n < 1000 or profile.get("must_explain")) else "gradient boosting"
    raise ValueError(f"unknown input {inp!r}")


# ---------------------------------------------------------------- Task 4
def diagnose(train_score: float, val_score: float, prod_score: float | None = None,
             floor: float = 0.7, gap: float = 0.05) -> str:
    """The debugging flowchart (Figure 24.7) as a function. Scores are "higher is better" (e.g. AUC).

    In this order:
      train_score < floor                                  -> "underfit"
      train_score - val_score > gap                        -> "overfit"
      prod_score is not None and val_score - prod_score > gap -> "leakage-or-drift"
      otherwise                                            -> "healthy"
    Example: diagnose(0.99, 0.80) -> "overfit"; diagnose(0.90, 0.88, 0.70) -> "leakage-or-drift"
    """
    # Approach: four ifs in the documented order
    if train_score < floor:
        return "underfit"
    if train_score - val_score > gap:
        return "overfit"
    if prod_score is not None and val_score - prod_score > gap:
        return "leakage-or-drift"
    return "healthy"


# ---------------------------------------------------------------- Task 5
def leakage_checklist(df: pd.DataFrame, target: str, as_of=None) -> dict:
    """Scan a raw table for the four mechanical leaks of the Section 2 checklist.

    Return {"id_like": [...], "constant": [...], "target_duplicate": [...], "future_dated": [...]}
    (column names, in df column order; the target itself is never listed):
      id_like:          integer or string columns whose number of unique values equals len(df)
      constant:         columns with <= 1 unique value (NaN counts as a value)
      target_duplicate: numeric columns with |corr(col, target)| > 0.999, OR non-numeric columns with
                        at most 20 unique values where every value maps to a single target value
                        (groupby(col)[target].nunique().max() == 1)
      future_dated:     datetime columns with any value later than ``as_of`` (pd.Timestamp); [] if as_of is None
    """
    # Approach: loop over columns once; pd.api.types.is_* helpers; keep four lists
    out = {"id_like": [], "constant": [], "target_duplicate": [], "future_dated": []}
    n = len(df)
    y = df[target]
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        if s.nunique(dropna=False) <= 1:
            out["constant"].append(col)
            continue
        if (pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object) \
                and s.nunique(dropna=True) == n:
            out["id_like"].append(col)
            continue
        if is_dt:
            if as_of is not None and (s > pd.Timestamp(as_of)).any():
                out["future_dated"].append(col)
            continue
        if is_num:
            if abs(s.corr(y.astype(float))) > 0.999:
                out["target_duplicate"].append(col)
        elif s.nunique(dropna=True) <= 20 and df.groupby(col)[target].nunique().max() == 1:
            out["target_duplicate"].append(col)
    return out


# ---------------------------------------------------------------- Task 6
def beat_the_baseline_by(margin: float = 0.05, seed: int = 0) -> dict:
    """On load_breast_cancer, beat a decision stump by ``margin`` ROC-AUC on unseen rows — leak-free.

    Protocol: train_test_split(test_size=0.25, stratify=y, random_state=seed). Baseline =
    DecisionTreeClassifier(max_depth=1, random_state=seed). Your model = Pipeline([("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))]). Both: 5-fold StratifiedKFold(shuffle=True, random_state=seed)
    cross_val_score(scoring="roc_auc") on the TRAINING rows only, then fit on the training rows and score
    ONCE on the test rows with roc_auc_score on predict_proba[:, 1]. Return
      {"baseline_cv", "model_cv", "baseline_test", "model_test", "model": fitted Pipeline,
       "n_train", "n_test", "beat_by_margin": model_test >= baseline_test + margin}
    The scaler must see only training rows (the tests check n_samples_seen_).
    """
    # Approach: follow the docstring literally; cross_val_score on X_tr only; fit on X_tr; roc_auc_score on X_te
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    baseline = DecisionTreeClassifier(max_depth=1, random_state=seed)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    baseline_cv = float(cross_val_score(baseline, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
    model_cv = float(cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
    baseline.fit(X_tr, y_tr)
    model.fit(X_tr, y_tr)
    baseline_test = float(roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1]))
    model_test = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    return {"baseline_cv": baseline_cv, "model_cv": model_cv, "baseline_test": baseline_test,
            "model_test": model_test, "model": model, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "beat_by_margin": bool(model_test >= baseline_test + margin)}


if __name__ == "__main__":
    print("[1] confusion matrix tp=43 fp=1 fn=9 tn=447 ->", {k: round(v, 3) for k, v in metrics_from_confusion(43, 1, 9, 447).items()})

    print("\n[2] which metric?")
    for p in [{"task": "binary", "positive_rate": 0.01}, {"task": "regression", "outliers": True},
              {"task": "binary", "positive_rate": 0.5, "probabilities_matter": True}, {"task": "forecasting"}]:
        print(f"   {p} -> {which_metric(p)}")

    print("\n[3] which model first?")
    for p in [{"input": "tabular", "n_rows": 400, "labeled": True}, {"input": "tabular", "n_rows": 200_000, "labeled": True},
              {"input": "images"}, {"input": "text", "n_rows": 5_000}, {"input": "tabular", "labeled": False, "goal": "anomaly"}]:
        print(f"   {p} -> {which_model_first(p)}")

    print("\n[4] diagnose:")
    for args in [(0.55, 0.53), (0.99, 0.80), (0.90, 0.88, 0.70), (0.90, 0.88, 0.87)]:
        print(f"   {args} -> {diagnose(*args)}")

    print("\n[5] leakage checklist on a planted table:")
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    demo = pd.DataFrame({"customer_id": np.arange(n), "country": "NO", "spend": rng.gamma(2, 30, n).round(1),
                         "label_copy": y * 10.0, "reason": np.where(y == 1, "left", "stayed"),
                         "cancel_date": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(-400, 200, n), "D"),
                         "churned": y})
    print("  ", leakage_checklist(demo, "churned", as_of="2026-01-01"))

    print("\n[6] beat the stump by 0.05 ROC-AUC:")
    res = beat_the_baseline_by(0.05)
    print(f"   baseline cv {res['baseline_cv']:.3f} test {res['baseline_test']:.3f} | "
          f"model cv {res['model_cv']:.3f} test {res['model_test']:.3f} | beat by margin: {res['beat_by_margin']}")
