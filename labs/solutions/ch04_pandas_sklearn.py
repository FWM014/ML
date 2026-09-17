"""
Lab 04 — From CSV to a saved model (Chapter 4: The Python Toolkit)
==================================================================
Chapter: 04 — The Python Toolkit

THE PROBLEM
-----------
Two jobs land on your desk on the same Monday. First, the operations lead at Kettle & Co
(an online kitchenware shop) exports `orders.csv` from the shop system and asks three
questions before the Thursday budget meeting: which region brings the most revenue,
whether gold-tier customers really spend more, and whether orders cluster on weekends.
The export has the usual problems: the `amount` column contains "n/a" strings, so pandas
reads it as text, and a few customers in the orders are missing from the CRM table.
Second, a partner clinic wants the tumour classifier from the chapter's "first complete
project" packaged so a nurse's laptop can load it and score one patient at a time. That
means a Pipeline (scaler + model) trained on training rows only, saved with joblib, and
a short table of the features that drive it for the clinical lead's slide.

TASKS (make the tests in labs/tests/test_ch04.py pass, one at a time)
---------------------------------------------------------------------
1. load_orders(path)                          → read the CSV with proper dtypes (dates, numbers)
2. revenue_by_region(orders)                  → groupby + named aggregation, sorted by revenue
3. attach_customer_tier(orders, customers)    → left merge; unmatched customers -> "unknown"
4. add_time_features(orders)                  → month, weekday, is_weekend, days_since_first
5. train_tumour_model(seed)                   → Pipeline(StandardScaler, LogisticRegression) on unseen rows
6. save_and_reload(model, path)               → joblib round trip; predictions identical
7. top_k_coefficients(model, feature_names, k)→ the k largest |coef| with sign, as a table

STRETCH (no tests)
------------------
- Time revenue_by_region on a 2-million-row orders frame built with np.repeat; then write the
  same aggregation with a Python loop over rows and compare (Section 2, "never loop over rows").
- Swap LogisticRegression for KNeighborsClassifier(n_neighbors=5), remove the scaler, and
  explain the drop in test accuracy (chapter exercise 2).

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "solutions":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def make_orders_csv(path: str | None = None, seed: int = 0, n: int = 400) -> str:
    """Write the Kettle & Co export to labs/data/ch04_orders.csv and return its path. Do not modify.

    Dirt: a few "n/a" strings in `amount` (the column will read as text), dates as strings.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = path or os.path.join(DATA_DIR, "ch04_orders.csv")
    amount = rng.gamma(3, 30, n).round(2).astype(object)
    amount[rng.choice(n, 6, replace=False)] = "n/a"
    orders = pd.DataFrame({
        "order_id": np.arange(10001, 10001 + n),
        "customer_id": rng.integers(1, 121, n),
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "amount": amount,
        "ordered": (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), "D")).strftime("%Y-%m-%d"),
    })
    orders.to_csv(path, index=False)
    return path


def make_customers(seed: int = 0) -> pd.DataFrame:
    """The CRM table: customers 1..110 only (ids 111-120 are missing). Do not modify."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 111)
    return pd.DataFrame({"customer_id": ids,
                         "tier": rng.choice(["gold", "silver", "bronze"], len(ids), p=[0.2, 0.3, 0.5])})


# ---------------------------------------------------------------- Task 1
def load_orders(path: str) -> pd.DataFrame:
    """Read the export with the right dtypes (chapter §3 "reading data", §4 gotcha 2).

    - `ordered` must become datetime64 (parse_dates).
    - `amount` must become float64: the "n/a" strings become NaN (pd.to_numeric(errors="coerce")).
    Returns the DataFrame with the same five columns.

    Example: a CSV row `10001,7,north,n/a,2024-03-02` -> amount NaN, ordered Timestamp('2024-03-02')
    """
    orders = pd.read_csv(path, parse_dates=["ordered"])
    orders["amount"] = pd.to_numeric(orders["amount"], errors="coerce")
    return orders


# ---------------------------------------------------------------- Task 2
def revenue_by_region(orders: pd.DataFrame) -> pd.DataFrame:
    """Which region brings the money? (chapter §3 groupby, named aggregation).

    Returns a DataFrame indexed by region with columns
        n_orders  (count of rows, NaN amounts included)
        revenue   (sum of amount, NaN skipped)
        avg_amount(mean of amount, NaN skipped)
    sorted by revenue, largest first.

    Example: two north orders of 10 and NaN, one south order of 5 ->
            n_orders  revenue  avg_amount
    north          2     10.0        10.0
    south          1      5.0         5.0
    """
    out = orders.groupby("region").agg(n_orders=("order_id", "size"),
                                       revenue=("amount", "sum"),
                                       avg_amount=("amount", "mean"))
    return out.sort_values("revenue", ascending=False)


# ---------------------------------------------------------------- Task 3
def attach_customer_tier(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Left-join the CRM tier onto every order (chapter §3 merge).

    Every order row is kept (how="left"); orders whose customer is not in the CRM get the
    tier "unknown" instead of NaN. Returns a new DataFrame with one extra column `tier`.

    Example: orders for customers [1, 999], customers table has 1 -> tiers [<tier of 1>, "unknown"]
    """
    joined = pd.merge(orders, customers[["customer_id", "tier"]], on="customer_id", how="left")
    joined["tier"] = joined["tier"].fillna("unknown")
    return joined


# ---------------------------------------------------------------- Task 4
def add_time_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the datetime column, vectorised (chapter §3 `.dt`).

    Adds four columns and returns a new DataFrame:
        month            1..12                      (ordered.dt.month)
        weekday          0=Monday .. 6=Sunday        (ordered.dt.weekday)
        is_weekend       True for Saturday/Sunday
        days_since_first integer days since the earliest order in the frame (0 for that order)

    Example: ordered = 2024-01-06 (a Saturday), earliest order 2024-01-01 ->
             month 1, weekday 5, is_weekend True, days_since_first 5
    """
    out = orders.copy()
    out["month"] = out["ordered"].dt.month
    out["weekday"] = out["ordered"].dt.weekday
    out["is_weekend"] = out["weekday"] >= 5
    out["days_since_first"] = (out["ordered"] - out["ordered"].min()).dt.days
    return out


# ---------------------------------------------------------------- Task 5
def train_tumour_model(seed: int = 42) -> dict:
    """The chapter's "first complete project" (§7): load -> split -> Pipeline -> fit -> evaluate.

    Steps: load_breast_cancer(); train_test_split(test_size=0.2, random_state=seed, stratify=y);
    Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]) fitted
    on the TRAINING rows only.
    Returns {"model": <fitted Pipeline>, "test_accuracy": float,
             "confusion_matrix": <2x2 ndarray from sklearn.metrics.confusion_matrix>,
             "feature_names": list[str]}.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {"model": model, "test_accuracy": float(accuracy_score(y_te, pred)),
            "confusion_matrix": confusion_matrix(y_te, pred), "feature_names": list(data.feature_names)}


# ---------------------------------------------------------------- Task 6
def save_and_reload(model, path: str):
    """Persist the WHOLE fitted pipeline with joblib and load it back (chapter §6 and §7).

    Creates the parent directory if needed, joblib.dump(model, path), then returns
    joblib.load(path). The returned object must predict exactly like the original.
    """
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    return joblib.load(path)


# ---------------------------------------------------------------- Task 7
def top_k_coefficients(model, feature_names: list[str], k: int = 5) -> pd.DataFrame:
    """The k features with the largest |coefficient| in the pipeline's "clf" step (chapter §6).

    The coefficients live in model.named_steps["clf"].coef_ (shape (1, n_features) for a
    binary LogisticRegression). Returns a DataFrame with columns ["feature", "coef"] and k
    rows sorted by |coef| descending, index reset to 0..k-1. Keep the sign of coef.

    Example: coef_ = [[0.5, -2.0, 1.0]], names ["a", "b", "c"], k=2 ->
        feature  coef
    0         b  -2.0
    1         c   1.0
    """
    coef = np.asarray(model.named_steps["clf"].coef_).ravel()
    table = pd.DataFrame({"feature": list(feature_names), "coef": coef})
    table["abs"] = table["coef"].abs()
    return (table.sort_values("abs", ascending=False, kind="stable")
                 .head(k).drop(columns="abs").reset_index(drop=True))


if __name__ == "__main__":
    path = make_orders_csv()
    orders = load_orders(path)
    print(orders.dtypes.to_string())
    print(f"\n{orders['amount'].isna().sum()} amounts unreadable -> NaN")
    print("\nrevenue by region:\n", revenue_by_region(orders).round(1).to_string())
    joined = attach_customer_tier(orders, make_customers())
    print("\nmean amount by tier:\n", joined.groupby("tier")["amount"].mean().round(1).to_string())
    feats = add_time_features(orders)
    print("\nweekend share of orders:", round(feats["is_weekend"].mean(), 3))

    res = train_tumour_model()
    print(f"\ntumour model test accuracy = {res['test_accuracy']:.3f}")
    print(res["confusion_matrix"])
    loaded = save_and_reload(res["model"], os.path.join(DATA_DIR, "ch04_tumour_model.joblib"))
    print("reloaded model predicts the same:", type(loaded).__name__)
    print("\ntop coefficients:\n", top_k_coefficients(res["model"], res["feature_names"], k=5).round(3).to_string())
