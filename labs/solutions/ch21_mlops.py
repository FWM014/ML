"""
Lab 21 — Shipping the churn model (Chapter: 21 — MLOps: Shipping & Monitoring Models)
=====================================================================================

THE PROBLEM
-----------
Meridian Telecom's churn model has lived in a notebook for six months. It is finally
going into production, but the serving box is locked down: no MLflow server, no web
framework, no internet — only Python, scikit-learn and the file system. Your job is
to build the minimal plumbing that Chapter 21 says every deployed model needs:
an experiment log you can query, a version-stamped model bundle that refuses to load
under the wrong library, a data bouncer that rejects bad batches, drift detection with
PSI and an alert policy, a training–serving skew check, a monitoring view that copes
with labels arriving 21 days late, and the request handler that the (future) web
layer will call. The training data are 3,000 customers with age, monthly spend, plan,
support calls and tenure; the label is ``churned``.

TASKS (make the tests in labs/tests/test_ch21.py pass, one at a time)
---------------------------------------------------------------------
1. log_run / best_run             → append-only JSON-lines experiment log and a query (Section 2)
2. save_bundle / load_bundle      → version-stamped joblib bundle; refuse a sklearn mismatch (Section 3)
3. psi / drift_alert              → Population Stability Index + per-column verdicts (Section 7)
4. validate_batch                 → schema / range / null / category checks that raise (Section 4)
5. detect_skew                    → training vs serving feature distributions (Section 5)
6. backfill_performance           → monitoring with delayed labels (Section 7)
7. predict_payload                → the pure-Python request handler with input validation (Section 4)

STRETCH
-------
- Add a KS test next to PSI in drift_alert and find a case where the two disagree.
- Write a "canary" function that routes 10% of a batch to a candidate bundle and reports
  both models' prediction distributions side by side.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE if os.path.basename(_HERE) == "labs" else os.path.dirname(_HERE), "data")

SCHEMA = {                                   # frozen with the model; the contract with callers
    "age":           {"dtype": "number", "min": 18, "max": 100, "nullable": False},
    "monthly_spend": {"dtype": "number", "min": 0, "max": 500, "nullable": False},
    "plan":          {"dtype": "category", "values": {"basic", "plus", "premium"}, "nullable": False},
    "support_calls": {"dtype": "number", "min": 0, "max": 50, "nullable": True},
    "tenure_months": {"dtype": "number", "min": 0, "max": 240, "nullable": False},
}
FEATURES = list(SCHEMA)


class DataValidationError(ValueError):
    """Raised when a batch or a payload violates the schema."""


def make_churn_data(seed: int = 0, n: int = 3000) -> pd.DataFrame:
    """Training data for the churn model (label column: churned). Do not modify."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "monthly_spend": rng.gamma(3, 15, n).round(2),
        "plan": rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2]),
        "support_calls": rng.poisson(1.2, n).astype(float),
        "tenure_months": rng.integers(0, 120, n).astype(float),
    })
    logit = (-1.2 + 0.5 * df["support_calls"] - 0.02 * df["tenure_months"] + 0.01 * (df["monthly_spend"] - 45)
             + np.where(df["plan"] == "basic", 0.6, np.where(df["plan"] == "plus", 0.0, -0.5)))
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def fit_churn_pipeline(df: pd.DataFrame, seed: int = 0):
    """A leak-free Pipeline (ColumnTransformer + LogisticRegression). Do not modify."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = ["age", "monthly_spend", "support_calls", "tenure_months"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
    ])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000, random_state=seed))])
    return pipe.fit(df[FEATURES], df["churned"])


def make_prediction_log(seed: int = 7, days: int = 90, n_per_day: int = 200, label_delay: int = 21,
                        drift_day: int = 45) -> pd.DataFrame:
    """Daily prediction log with labels that become known ``label_delay`` days later.

    From ``drift_day`` on, customers behave differently (CONCEPT drift): the inputs, and so the
    model's scores, look the same, but the true churn rate jumps. Columns: day, score, label,
    label_known_on. Do not modify.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        p_model = rng.beta(2, 5, n_per_day)
        score = np.clip(p_model + rng.normal(0, 0.12, n_per_day), 0, 1)
        shift = 0.0 if d < drift_day else 0.30
        label = (rng.random(n_per_day) < np.clip(p_model + shift, 0, 1)).astype(int)
        rows.append(pd.DataFrame({"day": d, "score": score, "label": label, "label_known_on": d + label_delay}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- Task 1
def log_run(log_path: str, name: str, params: dict, metrics: dict, tags: tuple = ()) -> str:
    """Append one JSON line describing a training run to ``log_path``; return its run_id.

    The record has keys run_id (10 hex chars: sha1 of name + time.time_ns()), ts (ISO string),
    name, params, metrics, sklearn (sklearn.__version__), tags (list). Create the parent
    directory if needed; never overwrite existing lines (open in append mode).
    """
    # Approach: build the dict, os.makedirs(dirname, exist_ok=True), open(path, "a") and write json.dumps(rec) + "\n"
    import sklearn

    rec = {"run_id": hashlib.sha1(f"{name}{time.time_ns()}".encode()).hexdigest()[:10],
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name, "params": dict(params), "metrics": dict(metrics),
           "sklearn": sklearn.__version__, "tags": list(tags)}
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec["run_id"]


def best_run(log_path: str, metric: str, higher_is_better: bool = True) -> dict:
    """Read the JSON-lines log back and return the full record with the best ``metric``.

    Runs that do not have ``metric`` in their metrics are ignored. Raises ValueError if no run has it.
    Example: best_run(path, "cv_auc") -> {"run_id": ..., "name": "rf_200", "metrics": {"cv_auc": 0.99, ...}, ...}
    """
    # Approach: read lines -> json.loads each; filter by metric key; max/min on rec["metrics"][metric]
    with open(log_path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if metric in r.get("metrics", {})]
    if not recs:
        raise ValueError(f"no run has metric {metric!r}")
    pick = max if higher_is_better else min
    return pick(recs, key=lambda r: r["metrics"][metric])


# ---------------------------------------------------------------- Task 2
def save_bundle(pipeline, path: str, model_version: str, extra: dict | None = None) -> str:
    """joblib.dump a dict bundle {"pipeline", "sklearn_version", "model_version", "features", **extra}.

    ``features`` is the module-level FEATURES list. Creates the parent directory. Returns ``path``.
    """
    # Approach: build the dict with sklearn.__version__, os.makedirs, joblib.dump(bundle, path)
    import joblib
    import sklearn

    bundle = {"pipeline": pipeline, "sklearn_version": sklearn.__version__, "model_version": model_version,
              "features": list(FEATURES), **(extra or {})}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str, strict: bool = True) -> dict:
    """joblib.load the bundle and check its sklearn_version against the runtime's.

    On a mismatch: raise RuntimeError(...) if strict, otherwise return the bundle with an extra key
    "warning" describing the mismatch. On a match return the bundle unchanged.
    """
    # Approach: joblib.load; compare bundle["sklearn_version"] with sklearn.__version__
    import joblib
    import sklearn

    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        msg = f"bundle pickled with sklearn {bundle.get('sklearn_version')}, runtime has {sklearn.__version__}"
        if strict:
            raise RuntimeError(msg)
        bundle = dict(bundle, warning=msg)
    return bundle


# ---------------------------------------------------------------- Task 3
def psi(reference: np.ndarray, current: np.ndarray, bins=10, eps: float = 1e-6) -> float:
    """Population Stability Index between a reference and a current sample.

    ``bins``: an int -> use that many quantile bins of the reference; or an array of bin edges.
    In both cases the outermost edges are replaced by -inf / +inf so nothing falls outside.
    PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%), each % padded with ``eps``.

    Example: reference = [1,1,2,2,3,3,4,4], current = [1,1,1,1,2,2,3,4],
             bins = [0.5, 1.5, 2.5, 3.5, 4.5]  ->  0.5 * ln 2 = 0.3466
    """
    # Approach: edges = np.quantile(reference, linspace) or np.asarray(bins); edges[0], edges[-1] = -inf, inf; np.histogram both
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if np.isscalar(bins):
        edges = np.quantile(reference, np.linspace(0, 1, int(bins) + 1))
    else:
        edges = np.asarray(bins, float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + eps
    cur_pct = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_alert(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10,
                investigate_at: float = 0.1, alert_at: float = 0.25) -> pd.DataFrame:
    """PSI per numeric column of ``reference`` with a verdict.

    Returns a DataFrame indexed by column with columns "psi" (float) and "verdict":
    "stable" if psi < investigate_at, "investigate" if psi < alert_at, else "alert".
    Non-numeric columns are skipped. Row order = column order of ``reference``.
    """
    # Approach: loop over reference.select_dtypes("number").columns, call psi, map to a verdict
    rows = []
    for col in reference.select_dtypes("number").columns:
        p = psi(reference[col].dropna().to_numpy(), current[col].dropna().to_numpy(), bins=bins)
        verdict = "stable" if p < investigate_at else ("investigate" if p < alert_at else "alert")
        rows.append({"column": col, "psi": p, "verdict": verdict})
    return pd.DataFrame(rows).set_index("column")


# ---------------------------------------------------------------- Task 4
def validate_batch(df: pd.DataFrame, schema: dict = SCHEMA, max_null_frac: float = 0.05,
                   max_out_of_range_frac: float = 0.01) -> bool:
    """The bouncer at the door. Raise DataValidationError listing EVERY problem, else return True.

    Checks, in this order: missing columns; unexpected columns; per column: null fraction above
    ``max_null_frac`` (or above 0 when nullable is False); numeric columns must have a numeric dtype
    and at most ``max_out_of_range_frac`` of rows outside [min, max]; category columns must contain
    only known values. Join the problem strings with "; " in the error message.
    """
    # Approach: collect problem strings in a list; set arithmetic on columns; pd.api.types.is_numeric_dtype; raise at the end
    problems = []
    missing = set(schema) - set(df.columns)
    extra = set(df.columns) - set(schema)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected columns: {sorted(extra)}")
    for col, rule in schema.items():
        if col not in df:
            continue
        s = df[col]
        null_frac = float(s.isna().mean())
        if null_frac > (max_null_frac if rule["nullable"] else 0):
            problems.append(f"{col}: {null_frac:.1%} nulls (nullable={rule['nullable']})")
        if rule["dtype"] == "number":
            if not pd.api.types.is_numeric_dtype(s):
                problems.append(f"{col}: expected numeric, got {s.dtype}")
                continue
            oor = float(((s < rule["min"]) | (s > rule["max"])).mean())
            if oor > max_out_of_range_frac:
                problems.append(f"{col}: {oor:.1%} of rows outside [{rule['min']}, {rule['max']}]")
        else:
            bad = set(s.dropna().unique()) - set(rule["values"])
            if bad:
                problems.append(f"{col}: unknown categories {sorted(map(str, bad))}")
    if problems:
        raise DataValidationError("; ".join(problems))
    return True


# ---------------------------------------------------------------- Task 5
def detect_skew(train_features: pd.DataFrame, serving_features: pd.DataFrame,
                psi_threshold: float = 0.1, alpha: float = 0.01) -> dict:
    """Compare the features the model was trained on with the features logged at serving time.

    Returns {"missing_in_serving": [...], "extra_in_serving": [...], "skewed": [...], "report": DataFrame}.
    For every numeric column present in both: psi (10 quantile bins of train) and the two-sample
    KS p-value (scipy.stats.ks_2samp). A column is "skewed" when psi > psi_threshold AND ks_p < alpha.
    The report is indexed by column with columns train_mean, serve_mean, psi, ks_p, skewed.
    """
    # Approach: set differences for the column lists; loop shared numeric columns; scipy.stats.ks_2samp
    from scipy import stats

    missing = sorted(set(train_features.columns) - set(serving_features.columns))
    extra = sorted(set(serving_features.columns) - set(train_features.columns))
    rows = []
    for col in train_features.select_dtypes("number").columns:
        if col not in serving_features:
            continue
        a, b = train_features[col].dropna().to_numpy(float), serving_features[col].dropna().to_numpy(float)
        p = psi(a, b, bins=10)
        ks_p = float(stats.ks_2samp(a, b).pvalue)
        rows.append({"column": col, "train_mean": float(a.mean()), "serve_mean": float(b.mean()),
                     "psi": p, "ks_p": ks_p, "skewed": bool(p > psi_threshold and ks_p < alpha)})
    report = pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
    skewed = list(report.index[report["skewed"]]) if len(report) else []
    return {"missing_in_serving": missing, "extra_in_serving": extra, "skewed": skewed, "report": report}


# ---------------------------------------------------------------- Task 6
def backfill_performance(log: pd.DataFrame, today: int, threshold: float = 0.5) -> dict:
    """What the monitoring dashboard can show on day ``today``.

    Only rows with day <= today exist yet. Return
      "pred_rate": Series indexed by day, mean(score > threshold) per day   (available immediately)
      "accuracy":  Series indexed by day, accuracy per day using ONLY rows whose label_known_on <= today
      "latest_labelled_day": int, the last day that has an accuracy value (or -1 if none)
    Accuracy = mean((score > threshold).astype(int) == label) within the day.
    """
    # Approach: seen = log[log.day <= today]; groupby("day") for the rate; filter label_known_on <= today for accuracy
    seen = log[log["day"] <= today]
    pred_rate = seen.groupby("day")["score"].apply(lambda s: float((s > threshold).mean()))
    labelled = seen[seen["label_known_on"] <= today]
    if len(labelled):
        accuracy = labelled.groupby("day").apply(
            lambda g: float(((g["score"] > threshold).astype(int) == g["label"]).mean()), include_groups=False)
        latest = int(accuracy.index.max())
    else:
        accuracy, latest = pd.Series(dtype=float), -1
    return {"pred_rate": pred_rate, "accuracy": accuracy, "latest_labelled_day": latest}


# ---------------------------------------------------------------- Task 7
def predict_payload(bundle: dict, payload: dict) -> dict:
    """The request handler the web layer will call. Pure Python, never raises on bad input.

    ``payload`` is one customer as a dict, e.g. {"customer_id": "c-17", "age": 40, "monthly_spend": 30.5,
    "plan": "basic", "support_calls": 1, "tenure_months": 12}. Validate against SCHEMA:
    every non-nullable feature must be present and non-null; numbers must be int/float (not bool)
    within [min, max]; categories must be known values. A nullable feature may be missing (-> NaN).
    Return
      {"ok": True, "customer_id": ..., "churn_probability": float, "model_version": bundle["model_version"]}
    or, on any problem,
      {"ok": False, "customer_id": ..., "errors": [<one string per problem>]}
    customer_id defaults to None if absent. Build a one-row DataFrame with the bundle's "features"
    in order and call bundle["pipeline"].predict_proba.
    """
    # Approach: loop over SCHEMA collecting errors; if errors return early; else pd.DataFrame([row], columns=features)
    errors = []
    row = {}
    for col, rule in SCHEMA.items():
        val = payload.get(col)
        if val is None:
            if rule["nullable"]:
                row[col] = np.nan
            else:
                errors.append(f"{col}: missing")
            continue
        if rule["dtype"] == "number":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errors.append(f"{col}: expected a number, got {type(val).__name__}")
                continue
            if not (rule["min"] <= val <= rule["max"]):
                errors.append(f"{col}: {val} outside [{rule['min']}, {rule['max']}]")
                continue
            row[col] = float(val)
        else:
            if val not in rule["values"]:
                errors.append(f"{col}: unknown category {val!r}")
                continue
            row[col] = val
    cid = payload.get("customer_id")
    if errors:
        return {"ok": False, "customer_id": cid, "errors": errors}
    X = pd.DataFrame([row], columns=bundle["features"])
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    return {"ok": True, "customer_id": cid, "churn_probability": p, "model_version": bundle["model_version"]}


if __name__ == "__main__":
    from sklearn.model_selection import cross_val_score

    df = make_churn_data()
    pipe = fit_churn_pipeline(df)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1] experiment log")
    log_path = os.path.join(DATA_DIR, "ch21_experiments_demo.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    for C in [0.01, 0.1, 1.0]:
        pipe.set_params(model__C=C)
        auc = cross_val_score(pipe, df[FEATURES], df["churned"], cv=5, scoring="roc_auc").mean()
        rid = log_run(log_path, f"logreg_C{C}", {"C": C}, {"cv_auc": round(float(auc), 4)}, tags=("sweep",))
        print(f"  {rid}  C={C:<5} cv_auc={auc:.4f}")
    b = best_run(log_path, "cv_auc")
    print(f"  best: {b['name']} ({b['run_id']})")
    os.remove(log_path)

    print("\n[2] bundle")
    pipe = fit_churn_pipeline(df)
    path = save_bundle(pipe, os.path.join(DATA_DIR, "ch21_churn_demo.joblib"), "churn-v1.0.0")
    bundle = load_bundle(path)
    print(f"  loaded {bundle['model_version']} built with sklearn {bundle['sklearn_version']}")

    print("\n[3] drift")
    rng = np.random.default_rng(1)
    ref = df[FEATURES].head(2000)
    cur = df[FEATURES].tail(1000).copy()
    cur["monthly_spend"] = cur["monthly_spend"] * 1.6
    print(drift_alert(ref, cur).round(3))

    print("\n[4] validation")
    print("  good batch:", validate_batch(df[FEATURES].head(50)))
    bad = df[FEATURES].head(50).copy(); bad.loc[bad.index[0], "age"] = 340; bad.loc[bad.index[1], "plan"] = "gold"
    try:
        validate_batch(bad)
    except DataValidationError as e:
        print("  bad batch rejected ->", e)

    print("\n[5] training-serving skew")
    serving = df[FEATURES].tail(800).copy()
    serving["tenure_months"] = serving["tenure_months"] / 12          # the serving code used YEARS
    sk = detect_skew(df[FEATURES].head(2000), serving)
    print("  skewed:", sk["skewed"])

    print("\n[6] delayed labels")
    log = make_prediction_log()
    for today in [44, 60, 89]:
        d = backfill_performance(log, today)
        print(f"  day {today}: pred-rate last 7 d = {d['pred_rate'].iloc[-7:].mean():.3f} | "
              f"accuracy known to day {d['latest_labelled_day']}, last 7 labelled days = {d['accuracy'].iloc[-7:].mean():.3f}")

    print("\n[7] request handler")
    print("  ", predict_payload(bundle, {"customer_id": "c-17", "age": 40, "monthly_spend": 30.5, "plan": "basic",
                                          "support_calls": 4, "tenure_months": 3}))
    print("  ", predict_payload(bundle, {"customer_id": "c-18", "age": 340, "monthly_spend": 30.5, "plan": "gold",
                                          "tenure_months": 3}))
    os.remove(path)
