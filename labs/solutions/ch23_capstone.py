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
    # Approach: literally fill in the dict; reread Section 1 of the chapter for what each slot means
    return {
        "problem": "Brightwave loses ~20% of subscribers a year; the retention offer is sent to everyone and loses money.",
        "decision": "Each month, which accounts receive the $20 retention offer (a ranked contact list with a cutoff).",
        "target": "churned = 1 if the account cancels within the next 30 days, else 0 (binary, ~20% positives).",
        "unit": "One active customer account per scoring month.",
        "prediction_time": "The first day of the month, using only data available at that moment.",
        "data": "4,000 accounts: usage numerics, plan, region, signup/last-login dates, NPS (5% missing); "
                "cancellation_reason is post-outcome and must be dropped.",
        "ranking_metric": "PR-AUC (average precision) with 5-fold stratified CV, ROC-AUC reported alongside.",
        "decision_metric": "Expected profit in $ per 1,000 customers = 0.4 * TP * $150 - flagged * $20, at the chosen threshold.",
        "baseline": "DummyClassifier(prior) and a logistic regression on the numeric columns; the model must beat both.",
        "cost_matrix": {"cost_offer": COST_OFFER, "value_saved": VALUE_SAVED, "save_rate": SAVE_RATE},
        "success_criterion": "Profit at the chosen threshold beats 'contact everyone' and 'contact nobody' on the test set, "
                             "with the PR-AUC 95% CI lower bound above the logistic baseline.",
        "constraints": "Batch scoring once a month; no features computed after the prediction date; explainable to marketing.",
        "risks": ["Leakage from post-outcome columns such as cancellation_reason",
                  "Offer effect (40% save rate) is an assumption; validate with a holdout in production",
                  "Drift when pricing changes"],
        "owner": "Retention marketing lead (business owner) and the ML engineer (model owner).",
    }


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
    # Approach: loop over columns, build a one-column numeric frame, cross_val_score(..., scoring="roc_auc")
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.tree import DecisionTreeClassifier

    n = len(df)
    y = df[target]
    missing = {c: float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()}
    id_like = [c for c in df.columns if c != target and df[c].nunique(dropna=True) == n]
    scores = {}
    for col in df.columns:
        if col == target or col in id_like:
            continue
        x = df[[col]]
        if pd.api.types.is_datetime64_any_dtype(x[col]):
            x = (x[col].astype("int64") // 10 ** 9).to_frame()
        elif not pd.api.types.is_numeric_dtype(x[col]):
            x = pd.DataFrame(OrdinalEncoder().fit_transform(x.astype(str)), columns=[col])
        x = x.fillna(-999)
        scores[col] = float(cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), x, y,
                                            cv=5, scoring="roc_auc").mean())
    ranked = pd.Series(scores).sort_values(ascending=False)
    return {"n_rows": int(n), "positive_rate": float(y.mean()), "missing_frac": missing, "id_like": id_like,
            "single_column_auc": ranked, "suspicious": list(ranked[ranked > auc_threshold].index)}


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
    # Approach: three inner Pipelines, ColumnTransformer([...]), outer Pipeline with steps "prep" and "model"
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), list(num_cols)),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), list(cat_cols)),
        ("date", Pipeline([("fx", FunctionTransformer(date_features, feature_names_out=date_feature_names)),
                           ("sc", StandardScaler())]), list(date_cols)),
    ])
    return Pipeline([("prep", prep), ("model", model)])


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
    # Approach: dict of candidates; loop cross_validate; collect rows; pd.DataFrame(...).sort_values
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate

    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed),
        "gbm": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed),
    }
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=seed)
    rows = []
    for name, m in candidates.items():
        r = cross_validate(build_pipeline(m), X_tr, y_tr, cv=cv, scoring=["average_precision", "roc_auc"])
        rows.append({"model": name, "pr_auc_mean": float(r["test_average_precision"].mean()),
                     "pr_auc_std": float(r["test_average_precision"].std()),
                     "roc_auc_mean": float(r["test_roc_auc"].mean()), "fit_seconds": float(r["fit_time"].mean())})
    return pd.DataFrame(rows).set_index("model").sort_values("pr_auc_mean", ascending=False)


# ---------------------------------------------------------------- Task 5
def tune(X_tr: pd.DataFrame, y_tr: pd.Series, n_iter: int = 6, seed: int = 0):
    """A small random search on the boosting pipeline, PR-AUC as the objective. Return the FITTED search.

    RandomizedSearchCV(build_pipeline(HistGradientBoostingClassifier(random_state=seed)), space, n_iter=n_iter,
    cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring="average_precision", random_state=seed)
    with space = {"model__max_iter": randint(60, 200), "model__max_depth": randint(2, 6),
                  "model__learning_rate": uniform(0.03, 0.15), "model__l2_regularization": uniform(0.0, 1.0)}
    (scipy.stats.randint / uniform).
    """
    # Approach: build the space dict, construct RandomizedSearchCV, .fit(X_tr, y_tr), return it
    from scipy.stats import randint, uniform
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    space = {"model__max_iter": randint(60, 200), "model__max_depth": randint(2, 6),
             "model__learning_rate": uniform(0.03, 0.15), "model__l2_regularization": uniform(0.0, 1.0)}
    search = RandomizedSearchCV(build_pipeline(HistGradientBoostingClassifier(random_state=seed)), space,
                                n_iter=n_iter, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                scoring="average_precision", random_state=seed)
    return search.fit(X_tr, y_tr)


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
    # Approach: a small profit(t) helper; np.array of profits over the grid; argmax
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    grid = np.arange(0.05, 0.95, 0.05) if grid is None else np.asarray(grid, float)

    def profit(t):
        flagged = proba >= t
        tp = (flagged & (y_true == 1)).sum()
        return float(save_rate * tp * value_saved - flagged.sum() * cost_offer)

    curve = pd.DataFrame({"threshold": grid, "flagged": [int((proba >= t).sum()) for t in grid],
                          "profit": [profit(t) for t in grid]})
    best = int(np.argmax(curve["profit"].to_numpy()))
    return {"threshold": float(grid[best]), "profit": float(curve["profit"].iloc[best]), "profit_curve": curve,
            "profit_everyone": profit(0.0), "profit_nobody": 0.0}


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
    # Approach: metrics from sklearn.metrics; bootstrap loop; permutation_importance(pipe, X_te, y_te, ...)
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(y_te)
    p = pipe.predict_proba(X_te)[:, 1]
    yhat = (p >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    boot = {"roc_auc": [], "pr_auc": []}
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        boot["roc_auc"].append(roc_auc_score(y[i], p[i]))
        boot["pr_auc"].append(average_precision_score(y[i], p[i]))
    ci = {k: tuple(float(v) for v in np.percentile(vals, [2.5, 97.5])) for k, vals in boot.items()}
    pi = permutation_importance(pipe, X_te, y_te, scoring="average_precision", n_repeats=5, random_state=seed)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False).head(5)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "precision": float(precision_score(y, yhat, zero_division=0)), "recall": float(recall_score(y, yhat)),
            "ci": ci, "top_features": [(str(c), float(v)) for c, v in imp.items()], "n_test": int(n)}


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
    # Approach: os.makedirs; joblib.dump; json.dump({**metadata, "sklearn": ..., "saved_at": ...}); reload and compare
    import joblib
    import sklearn

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{name}.joblib")
    meta_path = os.path.join(out_dir, f"{name}.json")
    joblib.dump(pipe, model_path)
    meta = {**metadata, "sklearn": sklearn.__version__, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    reloaded = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta2 = json.load(f)
    ok = True if X_check is None else bool(np.allclose(reloaded.predict_proba(X_check), pipe.predict_proba(X_check)))
    return {"model_path": model_path, "meta_path": meta_path, "meta": meta2, "reloaded": reloaded, "reload_ok": ok}


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
