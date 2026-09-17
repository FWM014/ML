"""
Lab 02 — The data audit (Chapter 2: Data: The Raw Material)
============================================================
Chapter: 02 — Data: The Raw Material

THE PROBLEM
-----------
Streamly, a subscription video service, wants a churn model and has handed you the
export from their billing system: about 1,240 customer rows with the label `churned`.
Retention will spend a €150k budget on whoever the model flags, so the Head of Data
wants an honest test score before anything is built. The export is exactly what real
exports look like: rows were appended twice by a retried job, an old ETL step writes
`age = -999` when the birth date is unknown, the `plan` column has been typed by three
different teams ("Premium ", "PLUS", " basic"), and someone helpfully joined in
`cancelled_date` from the cancellations table. Your job is the walk-around from
Section 10 of the chapter: profile, clean by transforming (not deleting), find the
columns that must never be features, and cut the data into sets the model can be
judged on. Only then does modelling start.

TASKS (make the tests in labs/tests/test_ch02.py pass, one at a time)
---------------------------------------------------------------------
1. class_balance(y)                        → counts, fractions and the majority baseline
2. missingness_report(df)                  → per-column missing count/fraction, worst first
3. fix_sentinels(df, column, sentinel)     → sentinel -> NaN plus a `<column>_missing` flag
4. normalise_and_dedupe(df, cat_columns)   → strip/lower categories, drop exact duplicates
5. find_suspect_columns(df, target)        → rule-based detection of ID and leaky columns
6. time_split(df, date_col, test_frac)     → sort by date, cut at a quantile, never shuffle
7. stratified_split(df, target, fracs)     → 60/20/20 train/val/test with equal churn rates

STRETCH (no tests)
------------------
- Add a `company_id` group column and write group_split() with GroupShuffleSplit; prove
  no company appears on both sides.
- Write a learning curve (Section 8) on the cleaned features with a logistic-regression
  pipeline and decide whether Streamly should buy more labelled rows.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_messy_customers(seed: int = 42, n: int = 1200, n_dups: int = 40) -> pd.DataFrame:
    """The Streamly export: n customers plus n_dups duplicated rows. Do not modify.

    Dirt injected on purpose: age sentinel -999, missing monthly_spend, inconsistent
    plan spelling, exact duplicate rows, and `cancelled_date` (only churners have one).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(500001, 500001 + n),
        "signup_date": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D"),
        "plan": rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2]),
        "age": rng.integers(18, 75, n),
        "tenure_months": rng.integers(1, 48, n),
        "monthly_spend": rng.gamma(4, 12, n).round(2),
        "support_calls": rng.poisson(1.2, n),
        "days_since_login": rng.integers(0, 90, n),
        "autopay": rng.random(n) < 0.6,
    })
    logit = (-2.6 + 0.035 * df.days_since_login + 0.5 * df.support_calls
             - 0.02 * df.tenure_months - 0.6 * df.autopay)
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # the leak: a cancellation date exists exactly for the customers who churned
    cancelled = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    churn_idx = df.index[df.churned == 1]
    cancelled[churn_idx] = (df.loc[churn_idx, "signup_date"]
                            + pd.to_timedelta(rng.integers(30, 400, len(churn_idx)), "D"))
    df["cancelled_date"] = cancelled
    # the dirt
    df.loc[rng.choice(n, 60, replace=False), "age"] = -999
    df.loc[rng.choice(n, 40, replace=False), "monthly_spend"] = np.nan
    messy_plan = rng.choice(n, 90, replace=False)
    df.loc[messy_plan[:30], "plan"] = "Premium "
    df.loc[messy_plan[30:60], "plan"] = "PLUS"
    df.loc[messy_plan[60:], "plan"] = " basic"
    df = pd.concat([df, df.sample(n_dups, random_state=1)])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- Task 1
def class_balance(y) -> dict:
    """Describe the label balance (chapter §6 and §10, row 11 of the audit checklist).

    Returns {"counts": {label: n}, "fractions": {label: share}, "majority_baseline": float}
    where majority_baseline is the accuracy of always predicting the most common label.

    Example: y = [0, 0, 0, 1] -> {"counts": {0: 3, 1: 1}, "fractions": {0: 0.75, 1: 0.25},
                                  "majority_baseline": 0.75}
    """
    s = pd.Series(np.asarray(y))
    counts = s.value_counts()
    fractions = counts / len(s)
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "fractions": {k: float(v) for k, v in fractions.items()},
        "majority_baseline": float(fractions.max()),
    }


# ---------------------------------------------------------------- Task 2
def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: how much is missing (chapter §6, EDA recipe).

    Returns a DataFrame indexed by column name with columns ["n_missing", "frac_missing"],
    sorted by frac_missing descending (ties: keep the original column order).
    Only NaN/NaT count as missing here: a -999 sentinel is NOT missing yet (see Task 3).

    Example: a 100-row frame where only "spend" has 5 NaN ->
        n_missing  frac_missing
    spend        5          0.05
    other        0          0.00
    """
    n_missing = df.isna().sum()
    report = pd.DataFrame({"n_missing": n_missing.astype(int), "frac_missing": n_missing / len(df)})
    return report.sort_values("frac_missing", ascending=False, kind="stable")


# ---------------------------------------------------------------- Task 3
def fix_sentinels(df: pd.DataFrame, column: str, sentinel: float = -999) -> pd.DataFrame:
    """Turn a sentinel value into an explicit NaN and keep the fact that it was missing (§7).

    Returns a NEW DataFrame (do not modify the input) in which
      - df[column] == sentinel has become NaN, and
      - a new integer column f"{column}_missing" is 1 where the value is now NaN, else 0.

    Example: column "age" = [34, -999, 51] -> age = [34.0, NaN, 51.0], age_missing = [0, 1, 0]
    """
    out = df.copy()
    out[column] = out[column].where(out[column] != sentinel)
    out[f"{column}_missing"] = out[column].isna().astype(int)
    return out


# ---------------------------------------------------------------- Task 4
def normalise_and_dedupe(df: pd.DataFrame, cat_columns: list[str]) -> pd.DataFrame:
    """Normalise category spelling, then remove exact duplicate rows (§7, "before splitting").

    For each column in cat_columns apply .str.strip().str.lower(). Then drop rows that are
    exact duplicates (keep the first) and reset the index. Returns a new DataFrame.

    Example: plan = ["Premium ", "premium", "PLUS"] -> ["premium", "premium", "plus"]; if two
    rows are now identical in every column, one of them is dropped.
    """
    out = df.copy()
    for c in cat_columns:
        out[c] = out[c].str.strip().str.lower()
    return out.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------- Task 5
def find_suspect_columns(df: pd.DataFrame, target: str, id_ratio: float = 0.95,
                         leak_corr: float = 0.9) -> dict:
    """Rule-based detection of columns that must not be features (§2 "the ID trap", §5 leakage).

    Rules (run this on the de-duplicated table):
      - ID: nunique / len(df) >= id_ratio and the dtype is not float (a unique value per row).
      - leaky: |corr| with the target >= leak_corr for EITHER the column's missingness
        indicator (notna as 0/1) OR its numeric values (numeric and bool columns only).
        The target itself is never reported.
    Returns {"id": [column names], "leaky": [column names]}, each in df column order.

    Example: a `cancelled_date` that is filled only for churners has a missingness indicator
    perfectly correlated with the label -> leaky. `customer_id` is unique per row -> id.
    """
    y = df[target].astype(float)
    ids, leaky = [], []
    for c in df.columns:
        if c == target:
            continue
        col = df[c]
        if col.nunique() / len(df) >= id_ratio and col.dtype.kind != "f":
            ids.append(c)
        candidates = [col.notna().astype(float)]
        if col.dtype.kind in "biuf":
            candidates.append(col.astype(float))
        for x in candidates:
            if x.nunique() < 2:          # a constant has no correlation
                continue
            r = x.corr(y)
            if pd.notna(r) and abs(r) >= leak_corr:
                leaky.append(c)
                break
    return {"id": ids, "leaky": leaky}


# ---------------------------------------------------------------- Task 6
def time_split(df: pd.DataFrame, date_col: str, test_frac: float = 0.2):
    """Time-based split (§4): sort by date, cut at the (1 - test_frac) quantile, never shuffle.

    Returns (train, test) DataFrames: train has date < cut, test has date >= cut. Every
    training row is dated strictly before every test row.

    Example: 100 rows dated day 1..100, test_frac=0.2 -> train = days 1..80, test = 81..100.
    """
    ordered = df.sort_values(date_col)
    cut = ordered[date_col].quantile(1 - test_frac)
    return ordered[ordered[date_col] < cut], ordered[ordered[date_col] >= cut]


# ---------------------------------------------------------------- Task 7
def stratified_split(df: pd.DataFrame, target: str, fracs=(0.6, 0.2, 0.2), seed: int = 0):
    """Stratified train / validation / test split done as two cuts (§4, the chapter's code).

    Cut 1: hold out fracs[2] of the rows as test, stratified on target, random_state=seed.
    Cut 2: from the remainder hold out fracs[1] / (fracs[0] + fracs[1]) as validation,
           stratified, same seed.
    Returns (train, val, test) DataFrames that keep df's original index, so the three
    index sets are disjoint and their union is df.index.

    Example: 1000 rows, 26% churn, fracs=(0.6, 0.2, 0.2) -> 600 / 200 / 200 rows and a churn
    rate of about 0.26 in each part.
    """
    from sklearn.model_selection import train_test_split

    trval, test = train_test_split(df, test_size=fracs[2], stratify=df[target], random_state=seed)
    val_share = fracs[1] / (fracs[0] + fracs[1])
    train, val = train_test_split(trval, test_size=val_share, stratify=trval[target], random_state=seed)
    return train, val, test


if __name__ == "__main__":
    raw = make_messy_customers()
    print(f"raw export: {raw.shape[0]} rows, {raw.shape[1]} columns")
    print("label balance:", class_balance(raw["churned"]))
    print("\nmissingness (before sentinel fix):")
    print(missingness_report(raw).head(4).to_string())
    fixed = fix_sentinels(raw, "age", -999)
    print(f"\nage: min before = {raw.age.min()}, after fix = {fixed.age.min()}, "
          f"{fixed.age_missing.sum()} rows flagged missing")
    clean = normalise_and_dedupe(fixed, ["plan"])
    print(f"plan levels: {sorted(clean.plan.unique())}; rows {len(fixed)} -> {len(clean)}")
    suspects = find_suspect_columns(clean, "churned")
    print("never features:", suspects)
    tr_t, te_t = time_split(clean, "signup_date", 0.2)
    print(f"time split: {len(tr_t)} train rows before {te_t.signup_date.min().date()}, {len(te_t)} test rows")
    train, val, test = stratified_split(clean, "churned")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s} n={len(part):4d}  churn rate={part.churned.mean():.3f}")
