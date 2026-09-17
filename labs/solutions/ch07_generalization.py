"""
Lab 07 — Will it generalise? (Chapter 7: Overfitting, Bias–Variance & Validation)
==================================================================================
Chapter: 07 — Overfitting, Bias–Variance & Validation

THE PROBLEM
-----------
You are the reviewer on a model-selection memo at Helio Diagnostics, a lab that wants
to ship a tumour classifier and a disease-progression regressor to partner clinics
next quarter. The memo's author picked a degree-12 polynomial "because it has the
lowest training error", scaled the whole dataset with StandardScaler before running
cross-validation, and proposes a deep decision tree with "100% training accuracy".
The CFO's only two questions are: which model do we ship, and do we spend €80k on
1,000 more labelled samples? Your review must produce, for each candidate, the two
numbers the chapter says settle every argument (cross-validated score and training
score), a diagnosis (high bias / high variance / ok), coefficient evidence that Ridge
and Lasso actually regularise, a validation curve over polynomial degree, a learning
curve that answers the data question, and a leak-free cross-validation protocol in
which every fitted step lives inside a Pipeline.

TASKS (make the tests in labs/tests/test_ch07.py pass, one at a time)
---------------------------------------------------------------------
1. diagnose(train_score, val_score, target, gap_tol)   → "high variance" / "high bias" / "ok"
2. degree_sweep(X, y, degrees, seed)                   → validation_curve; best degree by CV
3. ridge_shrinkage(X, y, alphas)                       → coefficients and their L2 norm per alpha
4. lasso_sparsity(X, y, alphas)                        → number of non-zero coefficients per alpha
5. cv_accuracy(X, y, C, seed)                          → StratifiedKFold CV with a Pipeline (no leak)
6. learning_curve_summary(model, X, y, sizes, seed)    → train/val scores by size + gap + "still rising"
7. model_report(X, y, seed)                            → CV score, train score, diagnosis per candidate

STRETCH (no tests)
------------------
- Nested CV: wrap GridSearchCV over ridge alpha inside cross_val_score and compare the honest
  R² with the (optimistic) best CV score from the inner grid.
- Reproduce the chapter's "selection outside CV" leak on pure noise with SelectKBest, then fix
  it by putting the selector inside the Pipeline.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np


def make_wave_data(seed: int = 0, n: int = 40):
    """40 noisy points of sin(2*pi*x) on [0, 1] (the chapter's degree sweep). Do not modify."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, n)[:, None]
    y = np.sin(2 * np.pi * X.ravel()) + rng.normal(0, 0.3, n)
    return X, y


# ---------------------------------------------------------------- Task 1
def diagnose(train_score: float, val_score: float, target: float = 0.95, gap_tol: float = 0.05) -> str:
    """The chapter's recipe (§4) as a rule. Scores are accuracy-like: higher is better.

    - gap = train_score - val_score. If gap > gap_tol -> "high variance" (fix variance first,
      even when bias is also present: the chapter's "both" row).
    - else if val_score < target -> "high bias" (both scores low, small gap).
    - else -> "ok" (ship it; stop tuning).

    Example: (0.99, 0.88) -> "high variance"; (0.86, 0.85) -> "high bias"; (0.995, 0.97) -> "ok"
    """
    if train_score - val_score > gap_tol:
        return "high variance"
    if val_score < target:
        return "high bias"
    return "ok"


# ---------------------------------------------------------------- Task 2
def degree_sweep(X: np.ndarray, y: np.ndarray, degrees=(1, 2, 3, 5, 9, 12), seed: int = 0) -> tuple[dict, int]:
    """validation_curve over polynomial degree (§2): choose by the VALIDATION column.

    model = make_pipeline(PolynomialFeatures(), StandardScaler(), LinearRegression());
    cv = KFold(5, shuffle=True, random_state=seed); param_name="polynomialfeatures__degree";
    scoring="neg_mean_squared_error". Average the folds and flip the sign.
    Returns ({degree: {"train_mse": ..., "val_mse": ...}}, best_degree) where best_degree has
    the lowest val_mse. Training MSE falls forever; validation MSE turns back up.

    Example: on make_wave_data() -> best_degree 3, val_mse at 12 > val_mse at 3.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, validation_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    model = make_pipeline(PolynomialFeatures(), StandardScaler(), LinearRegression())
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = validation_curve(model, X, y, param_name="polynomialfeatures__degree",
                              param_range=list(degrees), cv=cv, scoring="neg_mean_squared_error")
    results = {int(d): {"train_mse": float(-t), "val_mse": float(-v)}
               for d, t, v in zip(degrees, tr.mean(axis=1), va.mean(axis=1))}
    best = min(results, key=lambda d: results[d]["val_mse"])
    return results, best


# ---------------------------------------------------------------- Task 3
def ridge_shrinkage(X: np.ndarray, y: np.ndarray, alphas=(0.01, 1, 10, 100)) -> dict:
    """L2 shrinks every coefficient as alpha grows (§5 "the strength knob").

    For each alpha fit make_pipeline(StandardScaler(), Ridge(alpha=alpha)) on X, y (always
    scale before penalising) and read the Ridge step's coef_.
    Returns {alpha: {"coef": ndarray, "l2_norm": float}}. The l2_norm must strictly decrease
    as alpha increases.

    Example: on load_diabetes -> |w| ≈ 65 at alpha 0.01, ≈ 35 at alpha 100.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for a in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X, y)
        coef = model.named_steps["ridge"].coef_.copy()
        out[a] = {"coef": coef, "l2_norm": float(np.linalg.norm(coef))}
    return out


# ---------------------------------------------------------------- Task 4
def lasso_sparsity(X: np.ndarray, y: np.ndarray, alphas=(0.01, 1, 5, 20)) -> dict:
    """L1 switches features off (§5 "Lasso"): count the non-zero coefficients per alpha.

    For each alpha fit make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))
    and count (coef_ != 0).sum(). Returns {alpha: n_nonzero (int)}, non-increasing in alpha.

    Example: on load_diabetes -> {0.01: 10, 1: 7, 5: 5, 20: 3}
    """
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for a in alphas:
        model = make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=20000)).fit(X, y)
        out[a] = int((model.named_steps["lasso"].coef_ != 0).sum())
    return out


# ---------------------------------------------------------------- Task 5
def cv_accuracy(X: np.ndarray, y: np.ndarray, C: float = 0.1, seed: int = 0, n_splits: int = 5) -> dict:
    """Stratified k-fold accuracy of a leak-free logistic-regression pipeline (§7).

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000)): the scaler
    MUST be inside the pipeline so it is refit on each fold's training rows. Never call
    StandardScaler().fit_transform(X) on all rows first.
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed); scores via cross_val_score.
    Returns {"model": the (unfitted) pipeline, "scores": ndarray of fold accuracies,
             "mean": float, "std": float}.

    Example: on load_breast_cancer, C=0.1 -> mean ≈ 0.975 ± 0.012
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {"model": model, "scores": scores, "mean": float(scores.mean()), "std": float(scores.std())}


# ---------------------------------------------------------------- Task 6
def learning_curve_summary(model, X: np.ndarray, y: np.ndarray, sizes=(0.1, 0.25, 0.5, 0.75, 1.0),
                           seed: int = 0) -> dict:
    """Does more data help? (§4 learning curves, §6 "the best regularizer of all").

    learning_curve(model, X, y, train_sizes=sizes, cv=StratifiedKFold(5, shuffle=True,
    random_state=seed), scoring="accuracy", random_state=seed). Average over folds.
    Returns {"n_train": list[int], "train": list[float], "val": list[float],
             "gap_at_max": train[-1] - val[-1],
             "val_still_rising": bool, True when val[-1] - val[-2] > 0.005}.
    A large gap that keeps closing says variance (more data will help); a flat, early
    plateau with a small gap says bias (it will not).

    Example: a deep DecisionTreeClassifier on breast cancer -> train all 1.0, gap ≈ 0.07.
    """
    from sklearn.model_selection import StratifiedKFold, learning_curve

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n, tr, va = learning_curve(model, X, y, train_sizes=list(sizes), cv=cv, scoring="accuracy",
                               random_state=seed)
    train, val = tr.mean(axis=1), va.mean(axis=1)
    return {"n_train": [int(v) for v in n], "train": [float(v) for v in train], "val": [float(v) for v in val],
            "gap_at_max": float(train[-1] - val[-1]),
            "val_still_rising": bool(val[-1] - val[-2] > 0.005)}


# ---------------------------------------------------------------- Task 7
def model_report(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """The two numbers per candidate that end the meeting (§4 "before any modeling meeting").

    Candidates: {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
                 "deep_tree": DecisionTreeClassifier(random_state=seed)}.
    For each: cv_mean = mean accuracy over StratifiedKFold(5, shuffle=True, random_state=seed)
    (cross_val_score on a clone/unfitted model); train_score = accuracy after fitting on ALL rows
    and scoring the same rows; diagnosis = diagnose(train_score, cv_mean).
    Returns {"candidates": {name: {"cv_mean", "train_score", "diagnosis"}},
             "recommended": the name with the highest cv_mean}.
    """
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    candidates = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
                  "deep_tree": DecisionTreeClassifier(random_state=seed)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    report = {}
    for name, model in candidates.items():
        cv_mean = float(cross_val_score(clone(model), X, y, cv=cv, scoring="accuracy").mean())
        train_score = float(clone(model).fit(X, y).score(X, y))
        report[name] = {"cv_mean": cv_mean, "train_score": train_score,
                        "diagnosis": diagnose(train_score, cv_mean)}
    recommended = max(report, key=lambda k: report[k]["cv_mean"])
    return {"candidates": report, "recommended": recommended}


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.tree import DecisionTreeClassifier

    for tr_s, va_s in [(0.99, 0.88), (0.86, 0.85), (0.995, 0.97), (0.85, 0.70)]:
        print(f"train {tr_s:.3f} / val {va_s:.3f} -> {diagnose(tr_s, va_s)}")

    Xw, yw = make_wave_data()
    res, best = degree_sweep(Xw, yw)
    print("\ndegree  train MSE  valid MSE")
    for d, r in res.items():
        print(f"{d:6d}  {r['train_mse']:9.3f}  {r['val_mse']:9.3f}")
    print("best degree by validation:", best, "(noise floor 0.09)")

    Xd, yd = load_diabetes(return_X_y=True)
    print("\nRidge |w|2 by alpha:", {a: round(r["l2_norm"], 1) for a, r in ridge_shrinkage(Xd, yd).items()})
    print("Lasso non-zeros by alpha:", lasso_sparsity(Xd, yd))

    Xc, yc = load_breast_cancer(return_X_y=True)
    cv = cv_accuracy(Xc, yc, C=0.1)
    print(f"\n5-fold accuracy (pipeline): {cv['mean']:.3f} ± {cv['std']:.3f}")
    lc = learning_curve_summary(DecisionTreeClassifier(random_state=0), Xc, yc)
    print("deep tree learning curve: n =", lc["n_train"], "\n  train", np.round(lc["train"], 3),
          "\n  val  ", np.round(lc["val"], 3), f"\n  gap {lc['gap_at_max']:.3f}, still rising: {lc['val_still_rising']}")
    rep = model_report(Xc, yc)
    for name, r in rep["candidates"].items():
        print(f"{name:10s} CV {r['cv_mean']:.3f}  train {r['train_score']:.3f}  -> {r['diagnosis']}")
    print("recommended:", rep["recommended"])
