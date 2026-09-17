"""
Lab 20 — Explaining a credit model (Chapter: 20 — Explaining Models: SHAP & Friends)
====================================================================================

THE PROBLEM
-----------
Northgate Credit scores small personal loans with a gradient-boosted model. Three
people are waiting on you this week. The regulator wants to know *what the model
relies on* and whether the "free" impurity importances are trustworthy. A declined
customer, Ms. Okafor, wants to know *what she would have to change* to be approved.
Compliance wants numbers for a fairness review: the model never sees the protected
group attribute, but it does see ``zip_score``, a neighbourhood feature that is a
known proxy for it. You have 3,000 historical loans with the repayment outcome
(1 = repaid, 0 = defaulted) and the protected group for each applicant, which is used
only for auditing, never as a feature.

Everything in this lab is built by hand first, then checked against scikit-learn and
SHAP, so that you can defend every number in front of the regulator.

TASKS (make the tests in labs/tests/test_ch20.py pass, one at a time)
---------------------------------------------------------------------
1. permutation_importance_manual(model, X, y, n_repeats, seed) → mean log-loss increase per
   feature, from scratch; must match sklearn.inspection.permutation_importance (Section 4)
2. partial_dependence_manual(model, X, feature, grid) → the PDP curve as numbers; must match
   sklearn.inspection.partial_dependence (Section 5)
3. shapley_values(v, players) → exact Shapley values of a small cooperative game by enumerating
   every coalition (Section 6, "the fully worked example")
4. tree_shap_explain(rf, X_rows) → SHAP values with shap.TreeExplainer plus the efficiency
   check base + Σ shap = prediction (Section 6)
5. counterfactual(model, x_row, X_ref) → the smallest single-feature change (in std units)
   that flips a logistic-regression decision (Section 7)
6. fairness_metrics(y_true, y_pred, group) → selection rate and TPR per group, demographic-parity
   difference, equal-opportunity difference and the disparate-impact ratio (Section 9)

STRETCH
-------
- Post-process: find a per-group threshold that equalises TPR (equal opportunity) on the
  validation set and report what it does to the approval rate and FPR of each group.
- Fit a depth-2 surrogate tree to the GBM's predictions and report its fidelity; is it good
  enough to put in the model card?

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

FEATURES = ["income", "credit_score", "debt_ratio", "years_employed", "n_late_payments", "zip_score"]


def make_loan_data(seed: int = 0, n: int = 3000):
    """Synthetic loan book. Returns (X: DataFrame, y: repaid 0/1, group: protected attribute 0/1).

    Repayment depends on income, credit_score, debt_ratio and n_late_payments only.
    Group B (1) has lower income on average (historical inequity); zip_score proxies the group.
    The group is NOT a feature. Do not modify.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)
    income = rng.normal(52 - 9 * group, 13, n).clip(8, None)
    credit_score = rng.normal(0, 1, n)
    debt_ratio = rng.beta(2, 5, n)
    years_employed = rng.gamma(2, 3, n)
    n_late_payments = rng.poisson(0.8, n)
    zip_score = 0.8 * group + rng.normal(0, 0.5, n)
    logit = 0.06 * (income - 46) + 1.1 * credit_score - 2.5 * (debt_ratio - 0.3) - 0.5 * n_late_payments
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"income": income.round(2), "credit_score": credit_score.round(3),
                      "debt_ratio": debt_ratio.round(3), "years_employed": years_employed.round(1),
                      "n_late_payments": n_late_payments.astype(float), "zip_score": zip_score.round(3)})
    return X, y, group


def fit_gbm(X: pd.DataFrame, y: np.ndarray, seed: int = 0):
    """A small gradient-boosted classifier (the production model). Do not modify."""
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=seed).fit(X, y)


# ---------------------------------------------------------------- Task 1
def permutation_importance_manual(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 20,
                                  seed: int = 0) -> pd.Series:
    """Permutation importance from scratch: the mean increase in log-loss when one column is shuffled.

    For each column, ``n_repeats`` times: copy X, shuffle that column with
    ``rng = np.random.default_rng(seed)`` (one Generator for the whole call, ``rng.permutation``),
    recompute ``log_loss(y, model.predict_proba(X_shuffled)[:, 1])`` and record
    ``shuffled_loss - baseline_loss``. Return a Series indexed by X.columns with the MEAN increase,
    sorted descending.

    Example: a feature the model ignores gets ≈ 0; the most-used feature gets the largest value.
    """
    # Approach: baseline log-loss once; for each column loop n_repeats: X.copy(), column = rng.permutation(values), log-loss difference
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(seed)
    base = log_loss(y, model.predict_proba(X)[:, 1])
    out = {}
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(log_loss(y, model.predict_proba(Xp)[:, 1]) - base)
        out[col] = float(np.mean(deltas))
    return pd.Series(out).sort_values(ascending=False)


# ---------------------------------------------------------------- Task 2
def partial_dependence_manual(model, X: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    """Partial dependence of P(class 1) on one feature, computed by brute force.

    For each value g in ``grid``: copy X, set ``feature`` to g in EVERY row, average
    ``model.predict_proba(...)[:, 1]`` over the rows. Return a float array of len(grid).
    (Each row's curve is an ICE line; the PDP is their mean.)

    Example: grid = [10, 20, 30] -> array([0.31, 0.45, 0.60])
    """
    # Approach: loop over grid values, overwrite the column on a copy of X, average predict_proba[:, 1]
    vals = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        vals.append(float(model.predict_proba(Xg)[:, 1].mean()))
    return np.asarray(vals)


# ---------------------------------------------------------------- Task 3
def shapley_values(v: dict, players: list) -> dict:
    """Exact Shapley values of a cooperative game by enumerating every coalition.

    ``v`` maps a coalition (a *sorted tuple* of player names, ``()`` for the empty set) to its
    value. Use the weighted-coalition formula
        phi_i = sum over S not containing i of |S|! (n-|S|-1)! / n!  * [v(S ∪ {i}) - v(S)].
    Return {player: value}.

    Example (the taxi fare from the chapter):
        v = {(): 0, ("A",): 6, ("B",): 12, ("C",): 42, ("A","B"): 12, ("A","C"): 42,
             ("B","C"): 42, ("A","B","C"): 42}  ->  {"A": 2.0, "B": 5.0, "C": 35.0}
    Efficiency: the values always sum to v(all players).
    """
    # Approach: for each player, loop k in range(n) and S in combinations(others, k); weight = k!(n-k-1)!/n!
    n = len(players)

    def val(S):
        return v[tuple(sorted(S))]

    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for k in range(n):
            for S in combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                total += w * (val(list(S) + [p]) - val(S))
        phi[p] = float(total)
    return phi


# ---------------------------------------------------------------- Task 4
def tree_shap_explain(rf, X_rows: pd.DataFrame) -> dict:
    """Explain a fitted RandomForestClassifier on the given rows with shap.TreeExplainer.

    Return a dict with
      "base":       float, the explainer's expected value for class 1
      "shap":       array (n_rows, n_features) of SHAP values for class 1 (probability units)
      "prediction": array (n_rows,) = rf.predict_proba(X_rows)[:, 1]
      "efficiency_gap": float, max |base + shap.sum(axis=1) - prediction|  (should be ~ 0)
      "global_importance": Series mean |SHAP| per feature, sorted descending
    Note: for a binary classifier TreeExplainer returns one set of values per class — take
    class 1 (it may come back as a list of 2 arrays or as an array of shape (n, p, 2)).
    """
    # Approach: shap.TreeExplainer(rf); sv = explainer.shap_values(X_rows); pick class 1; expected_value[1]
    import shap

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_rows)
    ev = np.ravel(explainer.expected_value)
    if isinstance(sv, list):
        sv1 = np.asarray(sv[1])
    else:
        sv1 = np.asarray(sv)[:, :, 1] if np.asarray(sv).ndim == 3 else np.asarray(sv)
    base = float(ev[1] if len(ev) > 1 else ev[0])
    pred = rf.predict_proba(X_rows)[:, 1]
    gap = float(np.max(np.abs(base + sv1.sum(axis=1) - pred)))
    imp = pd.Series(np.abs(sv1).mean(axis=0), index=list(X_rows.columns)).sort_values(ascending=False)
    return {"base": base, "shap": sv1, "prediction": pred, "efficiency_gap": gap, "global_importance": imp}


# ---------------------------------------------------------------- Task 5
def counterfactual(model, x_row: pd.Series, X_ref: pd.DataFrame, target_class: int = 1,
                   max_std: float = 3.0, n_steps: int = 61) -> dict | None:
    """Smallest single-feature move (in std units of X_ref) that makes ``model`` predict target_class.

    For every feature f and every step in ``np.linspace(-max_std, max_std, n_steps)``:
    set f to x_row[f] + step * X_ref[f].std() (leave the other features untouched) and check
    ``model.predict(one_row_frame)[0] == target_class``. Keep the flip with the smallest |step|;
    on ties keep the earlier feature (column order). Return
    {"feature": f, "old": float, "new": float, "step_std": step} or None if nothing flips.

    Example: {"feature": "credit_score", "old": -0.4, "new": 0.6, "step_std": 1.0}
    """
    # Approach: nested loop over X_ref.columns and steps; build pd.DataFrame([x_row]) copies; keep min |step|
    steps = np.linspace(-max_std, max_std, n_steps)
    best = None
    for f in X_ref.columns:
        sd = float(X_ref[f].std())
        for step in steps:
            if best is not None and abs(step) >= abs(best["step_std"]):
                continue
            x = pd.DataFrame([x_row])
            x[f] = float(x_row[f]) + step * sd
            if int(model.predict(x)[0]) == target_class:
                best = {"feature": f, "old": float(x_row[f]), "new": float(x[f].iloc[0]), "step_std": float(step)}
    return best


# ---------------------------------------------------------------- Task 6
def fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> dict:
    """Group-fairness metrics for a binary decision (1 = approved).

    Return a dict with
      "selection_rate": {g: mean(y_pred == 1) within group g}
      "tpr":            {g: mean(y_pred == 1 | y_true == 1) within group g}   (equal opportunity)
      "demographic_parity_diff": max selection rate - min selection rate
      "equal_opportunity_diff":  max TPR - min TPR
      "disparate_impact_ratio":  min selection rate / max selection rate  (4/5 rule wants >= 0.8)
    Groups are the unique values of ``group`` (use plain ints as keys).

    Example: group A approves 75% and has TPR 1.0; group B approves 25% and has TPR 0.5 ->
      demographic_parity_diff 0.5, equal_opportunity_diff 0.5, disparate_impact_ratio 1/3
    """
    # Approach: loop over np.unique(group), boolean masks, then max/min of the two dicts
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    sel, tpr = {}, {}
    for g in np.unique(group):
        m = group == g
        sel[int(g)] = float((y_pred[m] == 1).mean())
        pos = m & (y_true == 1)
        tpr[int(g)] = float((y_pred[pos] == 1).mean()) if pos.any() else float("nan")
    s, t = np.array(list(sel.values())), np.array(list(tpr.values()))
    return {"selection_rate": sel, "tpr": tpr,
            "demographic_parity_diff": float(s.max() - s.min()),
            "equal_opportunity_diff": float(np.nanmax(t) - np.nanmin(t)),
            "disparate_impact_ratio": float(s.min() / s.max()) if s.max() > 0 else float("nan")}


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import partial_dependence, permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, group = make_loan_data()
    X_tr, X_va, y_tr, y_va, g_tr, g_va = train_test_split(X, y, group, test_size=0.3, random_state=0, stratify=y)
    gbm = fit_gbm(X_tr, y_tr)
    print(f"GBM validation accuracy = {gbm.score(X_va, y_va):.3f}")

    print("\n[1] permutation importance (log-loss increase), mine vs sklearn:")
    mine = permutation_importance_manual(gbm, X_va, y_va, n_repeats=10)
    sk = permutation_importance(gbm, X_va, y_va, scoring="neg_log_loss", n_repeats=10, random_state=0)
    for f, v in mine.items():
        print(f"  {f:<16}{v:7.3f}   sklearn {sk.importances_mean[list(X.columns).index(f)]:7.3f}")

    print("\n[2] partial dependence of P(repaid) on income:")
    grid = np.linspace(X["income"].quantile(0.05), X["income"].quantile(0.95), 6)
    print("  mine   :", partial_dependence_manual(gbm, X_va, "income", grid).round(3))
    pdp = partial_dependence(gbm, X_va, features=["income"], kind="average", custom_values={"income": grid},
                             method="brute", response_method="predict_proba")
    print("  sklearn:", np.round(pdp["average"][0], 3))

    print("\n[3] Shapley values for the taxi game:")
    v = {(): 0, ("A",): 6, ("B",): 12, ("C",): 42, ("A", "B"): 12, ("A", "C"): 42, ("B", "C"): 42, ("A", "B", "C"): 42}
    print("  ", shapley_values(v, ["A", "B", "C"]))

    print("\n[4] TreeExplainer on a random forest:")
    rf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=0).fit(X_tr, y_tr)
    ex = tree_shap_explain(rf, X_va.head(25))
    print(f"  base = {ex['base']:.3f}, efficiency gap = {ex['efficiency_gap']:.2e}")
    print("  global importance:", ex["global_importance"].round(3).to_dict())

    print("\n[5] counterfactual for a declined applicant (logistic regression):")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    declined = X_va[lr.predict(X_va) == 0].iloc[0]
    print("  ", counterfactual(lr, declined, X_tr))

    print("\n[6] fairness at threshold 0.5:")
    print("  ", fairness_metrics(y_va, lr.predict(X_va), g_va))
