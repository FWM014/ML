"""
Lab 10 — Forests, boosting and stacking at a lender (Chapter: 10 — Ensembles: Forests & Boosting)
==================================================================================================

THE PROBLEM
-----------
Harbor Lending approves about 3,000 personal loans a quarter and 20% of them default
within 12 months. The current scorecard is a logistic regression; the risk team wants
to know how much a tree ensemble adds before they commit to a tuning sprint and an
XGBoost/LightGBM dependency in the production stack. You have a 3,000-row table with
16 application features (income, debt-to-income, utilisation, late payments, credit
age, ...) plus a `row_id` column the data engineer forgot to drop. The questions on the
table: What does a random forest say the signal ceiling is (and which features does it
truly use)? How does boosting actually work — can you show it fixing residuals one
stump at a time? Where does a boosted model start to overfit, and what do XGBoost and
LightGBM report as their best iteration? Does stacking the scorecard with the ensembles
buy anything? And, for the sprint plan, in which order should the GBM knobs be tuned?

TASKS (make the tests in labs/tests/test_ch10.py pass, one at a time)
---------------------------------------------------------------------
1. forest_with_oob(X, y, n_estimators, seed)        → RandomForest with oob_score=True; return (rf, oob_accuracy)
2. top_features(rf, X_va, y_va, feature_names, k)   → top-k features by PERMUTATION importance on held-out rows
3. boost_from_scratch(x, y, n_rounds, lr)           → depth-1 gradient boosting on 1-D data; loss must fall every round
4. best_iteration_by_staging(X_tr, y_tr, X_va, y_va)→ GradientBoostingClassifier + staged_predict_proba → best n_estimators
5. boosters_with_early_stopping(...)                → XGBoost and LightGBM classifiers, early stopping, best iteration
6. stack_models(X_tr, y_tr, seed)                   → StackingClassifier(lr, rf, hgb) with out-of-fold predictions
7. tuning_plan(library)                             → the chapter's "order of operations": ordered parameter ranges

STRETCH (no tests)
------------------
* With sigma^2 = 1 compute Var = rho*sigma^2 + (1-rho)*sigma^2/B for rho in {0.6, 0.1} and
  B in {10, 1000}. Then lower max_features on your forest and measure the pairwise
  correlation of the trees' predictions — did rho move?
* Add scale_pos_weight=4 to the XGBoost model and compare AUC and mean predicted
  probability against the unweighted model (Pitfall 3).

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOAN_FEATURES = ["income", "dti", "utilization", "late_payments_24m", "months_oldest_account",
                 "employment_years", "loan_amount", "term_months", "interest_rate",
                 "num_open_accounts", "inquiries_6m", "revolving_balance", "delinq_2y",
                 "pub_rec", "annual_expenses", "credit_age_months"]


def make_loan_data(seed: int = 0, n: int = 3000) -> tuple[pd.DataFrame, np.ndarray]:
    """Loan applications (DataFrame with LOAN_FEATURES + row_id) and default label. Do not modify."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n, n_features=len(LOAN_FEATURES), n_informative=6,
                               n_redundant=3, n_repeated=0, flip_y=0.05, weights=[0.8, 0.2],
                               class_sep=0.9, random_state=seed)
    df = pd.DataFrame(X, columns=LOAN_FEATURES)
    df["row_id"] = np.arange(n)  # unique per row, carries no signal
    return df, y


def make_wave_data(seed: int = 0, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """One-dimensional noisy sine wave for the boosting-by-hand task. Do not modify."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 6, n))
    y = np.sin(x) + rng.normal(0, 0.2, n)
    return x, y


# ---------------------------------------------------------------- Task 1
def forest_with_oob(X, y, n_estimators: int = 200, seed: int = 0):
    """Fit RandomForestClassifier(n_estimators, max_features="sqrt", oob_score=True, random_state=seed).

    Returns (rf, oob_accuracy) where oob_accuracy is rf.oob_score_ (Section 3, "Out-of-bag
    error: your free validation set"). Use n_jobs=1 so results are identical everywhere.

    Example: on make_loan_data() the OOB accuracy is ~0.90 and lands within 0.03 of the
             accuracy on a held-out split.
    """
    # TODO: build the forest with oob_score=True, fit, return it with float(rf.oob_score_)
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt", oob_score=True,
                                n_jobs=1, random_state=seed).fit(X, y)
    return rf, float(rf.oob_score_)


# ---------------------------------------------------------------- Task 2
def top_features(rf, X_va, y_va, feature_names, k: int = 5, n_repeats: int = 10, seed: int = 0) -> list[str]:
    """Top-k feature names by permutation importance on HELD-OUT data (Section 3).

    Use sklearn.inspection.permutation_importance(rf, X_va, y_va, n_repeats=n_repeats,
    random_state=seed, scoring="roc_auc") and rank by importances_mean, descending.
    Do NOT use rf.feature_importances_ (impurity importance is biased toward
    high-cardinality columns such as row_id).

    Example: top_features(rf, X_va, y_va, LOAN_FEATURES + ["row_id"], k=5) never contains "row_id".
    """
    # TODO: permutation_importance -> argsort importances_mean descending -> first k names
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(rf, X_va, y_va, n_repeats=n_repeats, random_state=seed, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[::-1][:k]
    return [feature_names[i] for i in order]


# ---------------------------------------------------------------- Task 3
def boost_from_scratch(x: np.ndarray, y: np.ndarray, n_rounds: int = 50, learning_rate: float = 0.1):
    """Gradient boosting with squared error, by hand (Section 5), on 1-D data.

    F0 = mean(y). For m = 1..n_rounds:
        r = y - F(x)                                    (negative gradient of squared error)
        h = DecisionTreeRegressor(max_depth=1).fit(x, r)  (a stump)
        F(x) += learning_rate * h.predict(x)
        record train MSE of F after this round
    Returns (predict, losses) where predict(x_new) evaluates F on new points (shape (n,))
    and losses is a list of n_rounds MSE values that never increases.

    Example: on make_wave_data() the MSE drops from ~0.5 to below 0.1 in 50 rounds.
    """
    # TODO: keep a list of stumps; predict = f0 + lr * sum(stump.predict(...)); append MSE each round
    from sklearn.tree import DecisionTreeRegressor

    x2 = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    f0 = float(y.mean())
    F = np.full(len(y), f0)
    stumps, losses = [], []
    for _ in range(n_rounds):
        r = y - F
        h = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x2, r)
        F = F + learning_rate * h.predict(x2)
        stumps.append(h)
        losses.append(float(np.mean((y - F) ** 2)))

    def predict(x_new) -> np.ndarray:
        xn = np.asarray(x_new, dtype=float).reshape(-1, 1)
        out = np.full(len(xn), f0)
        for h in stumps:
            out = out + learning_rate * h.predict(xn)
        return out

    return predict, losses


# ---------------------------------------------------------------- Task 4
def best_iteration_by_staging(X_tr, y_tr, X_va, y_va, n_estimators: int = 300,
                              learning_rate: float = 0.05, max_depth: int = 3, seed: int = 0):
    """Draw the validation curve of a GBM yourself (Section 5, "Early stopping").

    Fit GradientBoostingClassifier(n_estimators, learning_rate, max_depth, subsample=0.8,
    random_state=seed) on the training rows, then use staged_predict_proba(X_va) to
    compute the validation log_loss after 1, 2, ..., n_estimators trees.
    Returns (best_n, val_curve): best_n = argmin(val_curve) + 1, val_curve a list of floats.

    Example: on the loan data the curve bottoms out well before 300 trees and then rises.
    """
    # TODO: fit, loop over staged_predict_proba, log_loss on column 1, argmin + 1
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss

    gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=max_depth, subsample=0.8, random_state=seed)
    gbc.fit(X_tr, y_tr)
    val_curve = [float(log_loss(y_va, p[:, 1])) for p in gbc.staged_predict_proba(X_va)]
    best_n = int(np.argmin(val_curve)) + 1
    return best_n, val_curve


# ---------------------------------------------------------------- Task 5
def boosters_with_early_stopping(X_fit, y_fit, X_va, y_va, seed: int = 0) -> dict:
    """XGBoost and LightGBM with early stopping on a separate validation split (Section 6).

    xgb : XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
          early_stopping_rounds=20, random_state=seed, n_jobs=1)
          .fit(X_fit, y_fit, eval_set=[(X_va, y_va)], verbose=False)
    lgbm: LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, min_child_samples=10,
          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed, n_jobs=1)
          .fit(X_fit, y_fit, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    Returns {"xgb": {"model": ..., "best_iteration": int, "val_auc": float},
             "lgbm": {"model": ..., "best_iteration": int, "val_auc": float}}
    where best_iteration is clf.best_iteration (xgb) / model.best_iteration_ (lgbm) and
    val_auc the ROC-AUC of predict_proba on the validation rows.

    Example: both best iterations are between 1 and 100 and both val AUCs are > 0.85.
    """
    # TODO: fit both models exactly as documented and read their best-iteration attributes
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                            early_stopping_rounds=20, random_state=seed, n_jobs=1)
    clf.fit(X_fit, y_fit, eval_set=[(X_va, y_va)], verbose=False)
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               verbose=-1, random_state=seed, n_jobs=1)
    model.fit(X_fit, y_fit, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    return {
        "xgb": {"model": clf, "best_iteration": int(clf.best_iteration),
                "val_auc": float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))},
        "lgbm": {"model": model, "best_iteration": int(model.best_iteration_),
                 "val_auc": float(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))},
    }


# ---------------------------------------------------------------- Task 6
def stack_models(X_tr, y_tr, seed: int = 0):
    """StackingClassifier with out-of-fold base predictions (Section 8).

    Base learners (named exactly "lr", "rf", "hgb"):
      lr  = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
      rf  = RandomForestClassifier(n_estimators=150, random_state=seed)
      hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=seed)
    Meta-learner: LogisticRegression(); cv=5; stack_method="predict_proba".
    Returns the FITTED StackingClassifier.

    Example: stack.final_estimator_.coef_ has shape (1, 3) — one weight per base model.
    """
    # TODO: assemble the estimator list and StackingClassifier(..., cv=5, stack_method="predict_proba"), fit
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("rf", RandomForestClassifier(n_estimators=150, random_state=seed)),
            ("hgb", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=seed))]
    stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=5,
                               stack_method="predict_proba")
    return stack.fit(X_tr, y_tr)


# ---------------------------------------------------------------- Task 7
def tuning_plan(library: str = "xgboost") -> dict[str, tuple[float, float]]:
    """The GBM tuning order of operations (Section 7) as an ORDERED dict {param: (low, high)}.

    Insertion order is the tuning order:
      step 0  learning_rate            (0.05, 0.1)   fixed first; early stopping sets the tree count
      step 1  capacity                 xgboost: max_depth (3, 8)      lightgbm: num_leaves (15, 127)
      step 2  leaf guard               xgboost: min_child_weight (1, 100)  lightgbm: min_child_samples (1, 100)
      step 3  subsample (0.5, 1.0), colsample_bytree (0.5, 1.0)
      step 4  reg_lambda (0, 10), reg_alpha (0, 1)
    "n_estimators" must NOT appear: it is set by early stopping, never searched.
    Raises ValueError for an unknown library.

    Example: list(tuning_plan("lightgbm"))[:2] == ["learning_rate", "num_leaves"]
    """
    # TODO: build the dict in tuning order, branching on library for the capacity and leaf-guard names
    if library == "xgboost":
        capacity, guard = ("max_depth", (3, 8)), ("min_child_weight", (1, 100))
    elif library == "lightgbm":
        capacity, guard = ("num_leaves", (15, 127)), ("min_child_samples", (1, 100))
    else:
        raise ValueError(f"unknown library {library!r}")
    plan: dict[str, tuple[float, float]] = {"learning_rate": (0.05, 0.1)}
    plan[capacity[0]] = capacity[1]
    plan[guard[0]] = guard[1]
    plan["subsample"] = (0.5, 1.0)
    plan["colsample_bytree"] = (0.5, 1.0)
    plan["reg_lambda"] = (0, 10)
    plan["reg_alpha"] = (0, 1)
    return plan


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = make_loan_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    rf, oob = forest_with_oob(X_tr, y_tr)
    print(f"forest: OOB accuracy {oob:.3f} | test accuracy {rf.score(X_te, y_te):.3f}")
    names = LOAN_FEATURES + ["row_id"]
    mdi_rank = list(pd.Series(rf.feature_importances_, index=names).rank(ascending=False).astype(int).sort_values().index[:5])
    print("top-5 by impurity   :", mdi_rank)
    print("top-5 by permutation:", top_features(rf, X_te, y_te, names, k=5))

    xw, yw = make_wave_data()
    predict, losses = boost_from_scratch(xw, yw, n_rounds=50, learning_rate=0.1)
    print(f"boosting by hand: MSE round 1 = {losses[0]:.3f}, round 10 = {losses[9]:.3f}, round 50 = {losses[-1]:.3f}")

    X_fit, X_va, y_fit, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0, stratify=y_tr)
    best_n, curve = best_iteration_by_staging(X_fit, y_fit, X_va, y_va)
    print(f"sklearn GBM: best n_estimators = {best_n} (val log-loss {curve[best_n - 1]:.3f}; at 300 trees {curve[-1]:.3f})")
    res = boosters_with_early_stopping(X_fit, y_fit, X_va, y_va)
    for name, r in res.items():
        print(f"{name:5s}: best iteration {r['best_iteration']:3d}  val AUC {r['val_auc']:.3f}  "
              f"test AUC {roc_auc_score(y_te, r['model'].predict_proba(X_te)[:, 1]):.3f}")

    stack = stack_models(X_tr, y_tr)
    for name, est in stack.named_estimators_.items():
        print(f"{name:5s} test AUC = {roc_auc_score(y_te, est.predict_proba(X_te)[:, 1]):.4f}")
    print(f"stack test AUC = {roc_auc_score(y_te, stack.predict_proba(X_te)[:, 1]):.4f}; "
          f"meta weights on [lr, rf, hgb] = {stack.final_estimator_.coef_[0].round(2)}")
    print("tuning order (xgboost):", list(tuning_plan("xgboost")))
