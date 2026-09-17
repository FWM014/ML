"""
Lab 23 — The churn project, end to end (Chapter: 23 — The End-to-End Project Playbook)
======================================================================================

THE PROBLEM
-----------
Brightwave, a subscription streaming company with 4,000 active accounts in the export
you were given, loses about one customer in five every year. Retention can send a $20
offer that saves roughly 40% of the churners who receive it; a saved customer is worth
$150. Marketing has been sending the offer to everyone, and losing money on it. You are
the ML person. In one week you must deliver a model that ranks accounts by churn risk,
a threshold that turns the ranking into a profitable contact list, an honest test-set
estimate with confidence intervals, a two-sentence explanation of what drives the
model, and a saved artifact the platform team can load. The export has numeric usage
columns, two categorical columns, two date columns, a customer id, some missing NPS
scores — and one column that would not exist at prediction time.

TASKS (make the tests in labs/tests/test_ch23.py pass, one at a time)
---------------------------------------------------------------------
1. frame_problem()                      → the filled project canvas (Section 1)
2. audit_data(df)                       → missingness, id-like columns and the single-column leakage hunt (Block 2)
3. build_pipeline(model, ...)           → ColumnTransformer: numeric, categorical, date features (Block 4)
4. compare_models(X_tr, y_tr)           → dummy / logistic / random forest / boosting by CV PR-AUC (Blocks 3–4)
5. tune(X_tr, y_tr)                     → a small RandomizedSearchCV on the boosting pipeline (Block 5)
6. choose_threshold(y, proba, ...)      → the cost-matrix threshold on out-of-fold probabilities (Block 6)
7. final_report(pipe, X_te, y_te, thr)  → one look at the test set with bootstrap CIs + top features (Block 7)
8. save_artifacts(pipe, meta)           → joblib + metadata JSON in labs/data/, and prove the reload (Block 8)

STRETCH
-------
- Add SHAP on the transformed features to final_report and reconcile the two rankings.
- Write the five-slide summary in money (Section 6): nobody / everyone / current rule / model.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE if os.path.basename(_HERE) == "labs" else os.path.dirname(_HERE), "data")

TARGET = "churned"
ID_COL = "customer_id"
NUM_COLS = ["tenure_months", "monthly_charges", "support_tickets", "logins_30d", "avg_session_min",
            "late_payments", "discount_pct", "products_held", "nps_score", "days_since_contact"]
CAT_COLS = ["plan", "region"]
DATE_COLS = ["signup_date", "last_login"]
CANVAS_KEYS = ["problem", "decision", "target", "unit", "prediction_time", "data", "ranking_metric",
               "decision_metric", "baseline", "cost_matrix", "success_criterion", "constraints", "risks", "owner"]
COST_OFFER, VALUE_SAVED, SAVE_RATE = 20.0, 150.0, 0.40


def make_churn_dataset(seed: int = 42, n: int = 4000) -> pd.DataFrame:
    """The Brightwave export: numeric usage, categoricals, dates, an id, missing NPS and one leak. Do not modify."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n, n_features=10, n_informative=6, n_redundant=2, weights=[0.8, 0.2],
                               flip_y=0.03, class_sep=0.9, random_state=seed)
    df = pd.DataFrame(X, columns=NUM_COLS).round(3)
    rng = np.random.default_rng(seed)
    df.insert(0, ID_COL, [f"C{100000 + i}" for i in range(n)])
    df["plan"] = np.where(df["monthly_charges"] > 0.5, "premium", np.where(df["monthly_charges"] > -0.5, "standard", "basic"))
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    df["signup_date"] = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    df["last_login"] = df["signup_date"] + pd.to_timedelta(rng.integers(30, 900, n), unit="D")
    df[TARGET] = y
    df["cancellation_reason"] = np.where(y == 1, rng.choice(["price", "service", "moved"], n), "n/a")   # the leak
    df.loc[rng.random(n) < 0.05, "nps_score"] = np.nan
    return df


def date_features(d: pd.DataFrame) -> pd.DataFrame:
    """Dates -> numbers: account_age_days and signup_month. Module-level so the pipeline pickles. Do not modify."""
    d = d.apply(pd.to_datetime)
    return pd.DataFrame({"account_age_days": (d["last_login"] - d["signup_date"]).dt.days,
                         "signup_month": d["signup_date"].dt.month})


def date_feature_names(transformer, input_cols):
    """Names for FunctionTransformer(feature_names_out=...). Do not modify."""
    return ["account_age_days", "signup_month"]


def split_once(df: pd.DataFrame, seed: int = 42):
    """Drop the leak and the id, then ONE stratified split. Returns X_tr, X_te, y_tr, y_te. Do not modify."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[TARGET, ID_COL, "cancellation_reason"])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


# ---------------------------------------------------------------- Task 1
def frame_problem() -> dict:
    """The project canvas, filled in for Brightwave. Return a dict with exactly the CANVAS_KEYS.

    Prose entries (problem, decision, target, unit, prediction_time, data, baseline, success_criterion,
    constraints, owner) are non-empty strings of at least 15 characters. "ranking_metric" must name
    PR-AUC (average precision) — the churn rate is ~20%; "decision_metric" must be expressed in money
    (mention "profit" or "$"). "cost_matrix" is {"cost_offer": 20, "value_saved": 150, "save_rate": 0.4}.
    "risks" is a list of at least two strings. Write it as you would for a real kickoff — the tests
    check the structure, your future self checks the content.
    """
    # TODO: literally fill in the dict; reread Section 1 of the chapter for what each slot means
    raise NotImplementedError("Task: frame_problem")


# ---------------------------------------------------------------- Task 2
def audit_data(df: pd.DataFrame, target: str = TARGET, auc_threshold: float = 0.95) -> dict:
    """Look before you model: missingness, id-like columns and the single-column leakage hunt.

    Return
      "n_rows":            int
      "positive_rate":     float, mean of the target
      "missing_frac":      {column: fraction of NaN} for columns with ANY missing values
      "id_like":           columns (other than the target) whose number of unique values == n_rows
      "single_column_auc": Series (sorted descending) — for every non-id, non-target column, the mean 5-fold
                           CV ROC-AUC of DecisionTreeClassifier(max_depth=3, random_state=0) on that column
                           alone (datetimes -> int64 seconds; strings -> OrdinalEncoder; NaN -> -999)
      "suspicious":        columns whose single-column AUC > auc_threshold (a leak until proven otherwise)
    """
    # TODO: loop over columns, build a one-column numeric frame, cross_val_score(..., scoring="roc_auc")
    raise NotImplementedError("Task: audit_data")


# ---------------------------------------------------------------- Task 3
def build_pipeline(model, num_cols=NUM_COLS, cat_cols=CAT_COLS, date_cols=DATE_COLS):
    """Pipeline([("prep", ColumnTransformer), ("model", model)]) — all feature engineering inside.

    The ColumnTransformer has three named blocks:
      "num":  SimpleImputer(median) -> StandardScaler                                on num_cols
      "cat":  SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")  on cat_cols
      "date": FunctionTransformer(date_features, feature_names_out=date_feature_names) -> StandardScaler
              on date_cols
    Feature names out look like "num__tenure_months", "cat__plan_premium", "date__account_age_days".
    """
    # TODO: three inner Pipelines, ColumnTransformer([...]), outer Pipeline with steps "prep" and "model"
    raise NotImplementedError("Task: build_pipeline")


# ---------------------------------------------------------------- Task 4
def compare_models(X_tr: pd.DataFrame, y_tr: pd.Series, cv_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Beat the dummies: four candidates, the same stratified CV, ranked by PR-AUC.

    Candidates (index labels): "dummy" = DummyClassifier(strategy="prior"), "logreg" =
    LogisticRegression(max_iter=1000), "rf" = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
    random_state=seed), "gbm" = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
    random_state=seed) — each wrapped in build_pipeline. Use StratifiedKFold(cv_splits, shuffle=True,
    random_state=seed) and cross_validate with scoring ["average_precision", "roc_auc"].
    Return a DataFrame with columns pr_auc_mean, pr_auc_std, roc_auc_mean, fit_seconds, sorted by
    pr_auc_mean descending.
    """
    # TODO: dict of candidates; loop cross_validate; collect rows; pd.DataFrame(...).sort_values
    raise NotImplementedError("Task: compare_models")


# ---------------------------------------------------------------- Task 5
def tune(X_tr: pd.DataFrame, y_tr: pd.Series, n_iter: int = 6, seed: int = 0):
    """A small random search on the boosting pipeline, PR-AUC as the objective. Return the FITTED search.

    RandomizedSearchCV(build_pipeline(HistGradientBoostingClassifier(random_state=seed)), space, n_iter=n_iter,
    cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="average_precision", random_state=seed)
    with space = {"model__max_iter": randint(60, 200), "model__max_depth": randint(2, 6),
                  "model__learning_rate": uniform(0.03, 0.15), "model__l2_regularization": uniform(0.0, 1.0)}
    (scipy.stats.randint / uniform).
    """
    # TODO: build the space dict, construct RandomizedSearchCV, .fit(X_tr, y_tr), return it
    raise NotImplementedError("Task: tune")


# ---------------------------------------------------------------- Task 6
def choose_threshold(y_true, proba, cost_offer: float = COST_OFFER, value_saved: float = VALUE_SAVED,
                     save_rate: float = SAVE_RATE, grid=None) -> dict:
    """Turn probabilities into a contact list that maximises expected profit.

    For each threshold t in grid (default np.arange(0.05, 0.95, 0.05)): flagged = proba >= t,
    tp = flagged & (y_true == 1), profit(t) = save_rate * tp.sum() * value_saved - flagged.sum() * cost_offer.
    Return {"threshold": best t (first in grid order on ties), "profit": its profit,
            "profit_curve": DataFrame with columns threshold, flagged, profit,
            "profit_everyone": profit at threshold 0.0, "profit_nobody": 0.0}
    Example: y = [1,1,0,0,1,0], p = [0.9,0.6,0.55,0.3,0.2,0.1], grid = [0.15, 0.4, 0.5, 0.8]
             -> threshold 0.15, profit 80 (3 TP * 60 - 5 * 20)
    """
    # TODO: a small profit(t) helper; np.array of profits over the grid; argmax
    raise NotImplementedError("Task: choose_threshold")


# ---------------------------------------------------------------- Task 7
def final_report(pipe, X_te: pd.DataFrame, y_te, threshold: float, n_boot: int = 200, seed: int = 0) -> dict:
    """ONE look at the sealed test set, with uncertainty, then the explanation.

    p = pipe.predict_proba(X_te)[:, 1]; yhat = p >= threshold. Return
      "roc_auc", "pr_auc" (average_precision_score), "precision", "recall": point estimates
      "ci": {"roc_auc": (lo, hi), "pr_auc": (lo, hi)} — 2.5/97.5 percentiles over n_boot bootstrap
            resamples of the test rows (rng = np.random.default_rng(seed); rng.integers(0, n, n))
      "top_features": list of (column, importance) — the 5 largest permutation importances of the RAW
            columns (sklearn.inspection.permutation_importance on the whole pipeline,
            scoring="average_precision", n_repeats=5, random_state=seed), sorted descending
      "n_test": len(y_te)
    """
    # TODO: metrics from sklearn.metrics; bootstrap loop; permutation_importance(pipe, X_te, y_te, ...)
    raise NotImplementedError("Task: final_report")


# ---------------------------------------------------------------- Task 8
def save_artifacts(pipe, metadata: dict, out_dir: str = DATA_DIR, name: str = "ch23_churn_model",
                   X_check: pd.DataFrame | None = None) -> dict:
    """Persist the whole pipeline + a metadata JSON, then prove the reload.

    Writes <out_dir>/<name>.joblib (joblib.dump(pipe)) and <out_dir>/<name>.json containing
    ``metadata`` plus "sklearn": sklearn.__version__ and "saved_at": an ISO timestamp. Creates out_dir.
    Reload both; if X_check is given, compare predict_proba of the reloaded pipeline with the original.
    Return {"model_path", "meta_path", "meta": the reloaded dict, "reloaded": the reloaded pipeline,
            "reload_ok": bool (True when X_check is None or predictions match with np.allclose)}
    """
    # TODO: os.makedirs; joblib.dump; json.dump({**metadata, "sklearn": ..., "saved_at": ...}); reload and compare
    raise NotImplementedError("Task: save_artifacts")


if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    canvas = frame_problem()
    print("[1] canvas:", {k: (v if isinstance(v, (dict, list)) else v[:60] + "...") for k, v in list(canvas.items())[:4]})

    df = make_churn_dataset()
    audit = audit_data(df)
    print(f"\n[2] audit: {audit['n_rows']} rows, churn rate {audit['positive_rate']:.3f}, missing {audit['missing_frac']}")
    print("    id-like:", audit["id_like"], "| SUSPICIOUS:", audit["suspicious"])
    print(audit["single_column_auc"].round(3).head(4).to_string())

    X_tr, X_te, y_tr, y_te = split_once(df)
    print("\n[3] pipeline feature names:", list(build_pipeline(None)[:-1].fit(X_tr).get_feature_names_out())[:4], "...")

    print("\n[4] model comparison (CV on the training set):")
    print(compare_models(X_tr, y_tr).round(3))

    print("\n[5] random search:")
    search = tune(X_tr, y_tr)
    print(f"    best CV PR-AUC = {search.best_score_:.3f}  params = { {k.replace('model__', ''): round(float(v), 3) for k, v in search.best_params_.items()} }")

    print("\n[6] threshold from out-of-fold probabilities:")
    p_oof = cross_val_predict(search.best_estimator_, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    thr = choose_threshold(y_tr, p_oof)
    print(f"    threshold {thr['threshold']:.2f} -> profit ${thr['profit']:.0f} | everyone ${thr['profit_everyone']:.0f} | nobody $0")

    print("\n[7] test set, once:")
    rep = final_report(search.best_estimator_, X_te, y_te, thr["threshold"])
    print(f"    PR-AUC {rep['pr_auc']:.3f} CI {tuple(round(v, 3) for v in rep['ci']['pr_auc'])} | "
          f"ROC-AUC {rep['roc_auc']:.3f} | precision {rep['precision']:.2f} recall {rep['recall']:.2f}")
    print("    top features:", [(f, round(v, 3)) for f, v in rep["top_features"]])

    print("\n[8] artifacts:")
    art = save_artifacts(search.best_estimator_, {"threshold": thr["threshold"], "features": list(X_tr.columns),
                                                  "best_params": search.best_params_, "test_pr_auc": rep["pr_auc"]},
                         X_check=X_te)
    print("    saved", os.path.basename(art["model_path"]), "| reload ok:", art["reload_ok"])
    for pth in (art["model_path"], art["meta_path"]):
        os.remove(pth)
