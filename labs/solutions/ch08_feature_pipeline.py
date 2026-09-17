"""
Lab 08 — Feature engineering & pipelines (Chapter: 08 — Feature Engineering & Pipelines)
=========================================================================================

THE PROBLEM
-----------
Brightline Broadband has 3,000 subscribers and loses about 20% of them a year. The
retention team wants a churn score before each renewal call, and the scoring service
will receive the raw CRM extract exactly as it is: tenure in months, monthly spend
(right-skewed, a few corporate accounts paying 20x the median, 8% blank), data usage
(heavy-tailed), the share of months on a discount, support-call counts, plan tier,
region (5% blank), the customer's city (250 distinct values, and new ones appear every
month), and the sign-up timestamp. Nobody will re-implement your preprocessing in the
serving layer: whatever you build must be ONE scikit-learn object that survives blanks,
never-seen plan names, and a brand-new city without crashing — and whose validation
score you can defend, which means no leakage from the label into the features.

You also inherit twelve customer-feedback snippets that the support lead labelled by
hand (complaint vs praise) and wants classified automatically.

TASKS (make the tests in labs/tests/test_ch08.py pass, one at a time)
---------------------------------------------------------------------
1. choose_scaler(df, num_cols)                    → per-column scaler rule: minmax / robust / standard
2. log_transform_skewed(df, threshold)            → log1p on the skewed non-negative columns, report which
3. cyclical_encode(df, col, period)               → add <col>_sin and <col>_cos so 23:00 is next to 00:00
4. target_encode_cv(cat_tr, y_tr, cat_te, ...)    → smoothed, CROSS-FITTED target encoding that cannot leak
5. build_pipeline(num_cols, cat_cols, date_col)   → ColumnTransformer + Pipeline that survives NaN and unknowns
6. text_classifier(docs, labels)                  → TF-IDF (1-2 grams) + LogisticRegression on the feedback
7. coefficient_table(fitted_pipeline)             → get_feature_names_out() paired with the model's coefficients

STRETCH (no tests)
------------------
* Add a ratio branch (support_calls / tenure_months) through a FunctionTransformer and
  check whether the cross-validated AUC moves. Then swap the classifier for
  HistGradientBoostingClassifier and drop the scaler — does anything change? Why?
* Give `city` to the pipeline three ways — dropped, one-hot with min_frequency, and via
  your cross-fitted target encoder — and rank the three honest CV scores.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_COLS = ["tenure_months", "monthly_spend", "data_gb", "discount_share", "support_calls"]
CAT_COLS = ["plan", "region"]
DATE_COL = "signup_ts"

FEEDBACK = [
    "refund not received, very disappointed", "great product, fast delivery",
    "terrible support, still waiting for refund", "love it, works perfectly",
    "broken on arrival, want my money back", "excellent quality, highly recommend",
    "never again, awful experience", "five stars, would buy again",
    "delivery late and package damaged", "fantastic value, very happy",
    "worst purchase ever, waste of money", "perfect fit, great quality",
]
FEEDBACK_LABELS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = complaint, 1 = praise


def make_customer_data(seed: int = 42, n: int = 3000) -> pd.DataFrame:
    """Synthetic Brightline CRM extract with a `churned` label. Do not modify.

    Numeric columns with skew and outliers, two low-cardinality categoricals, one
    high-cardinality `city` column that carries real signal, blanks in
    monthly_spend and region, and a sign-up timestamp with an hour of day.
    """
    rng = np.random.default_rng(seed)
    n_cities = 250
    city_effect = rng.normal(0, 0.8, n_cities)  # hidden per-city churn propensity
    city_id = rng.integers(0, n_cities, n)
    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": np.round(rng.lognormal(3.5, 0.4, n), 2),
        "data_gb": np.round(rng.lognormal(1.5, 1.2, n), 1),
        "discount_share": np.round(rng.beta(2, 5, n), 3),
        "support_calls": rng.poisson(1.2, n),
        "plan": rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2]),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "city": np.array([f"city_{i:03d}" for i in range(n_cities)])[city_id],
        "signup_ts": (pd.Timestamp("2023-01-01")
                      + pd.to_timedelta(rng.integers(0, 365, n), "D")
                      + pd.to_timedelta(rng.integers(0, 24, n), "h")),
    })
    corporate = rng.random(n) < 0.005                      # a few corporate accounts
    df.loc[corporate, "monthly_spend"] *= 20
    spend_missing = rng.random(n) < 0.08
    df.loc[spend_missing, "monthly_spend"] = np.nan        # 8% blank (and blank = risk)
    df.loc[rng.random(n) < 0.05, "region"] = np.nan        # 5% blank
    logit = (-2.2 + 0.6 * df.support_calls - 0.03 * df.tenure_months
             + 0.8 * (df.plan == "basic") + 0.7 * (df.signup_ts.dt.month >= 10)
             + 0.5 * spend_missing + city_effect[city_id])
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def make_new_customers() -> pd.DataFrame:
    """Five rows as the scoring service will see them. Do not modify.

    Contains an unseen plan ("enterprise"), a blank spend, a blank region and a city
    that was never in the training extract.
    """
    return pd.DataFrame({
        "tenure_months": [3, 40, 12, 60, 1],
        "monthly_spend": [45.0, np.nan, 38.5, 120.0, 29.9],
        "data_gb": [2.0, 15.5, 0.4, 80.0, 3.1],
        "discount_share": [0.1, 0.0, 0.5, 0.2, 0.9],
        "support_calls": [4, 0, 1, 2, 5],
        "plan": ["basic", "enterprise", "plus", "premium", "basic"],
        "region": ["north", np.nan, "south", "east", "west"],
        "city": ["city_001", "city_999", "city_010", "new_town", "city_002"],
        "signup_ts": pd.to_datetime(["2023-11-02 23:00", "2023-02-14 09:00", "2023-06-30 00:00",
                                     "2023-12-24 17:00", "2024-01-05 13:00"]),
    })


# ---------------------------------------------------------------- Task 1
def choose_scaler(df: pd.DataFrame, num_cols: list[str]) -> dict[str, str]:
    """Pick a scaler per numeric column with the chapter's rules (Section 2).

    For each column (NaN ignored), in this order:
      * "minmax"   if the column is bounded in [0, 1] (min >= 0 and max <= 1);
      * "robust"   if the 99th percentile is more than 10x the median (heavy tail);
      * "standard" otherwise.
    Returns {column: scaler_name}.

    Example: {"share": [0.1, 0.5, 0.9], "amount": [1, 2, 3, 1000], "age": [30, 40, 50]}
             -> {"share": "minmax", "amount": "robust", "age": "standard"}
    """
    # TODO: for each column compute min, max, median, 99th percentile (nan-aware) and apply the rules in order
    out: dict[str, str] = {}
    for c in num_cols:
        x = df[c].dropna().to_numpy(dtype=float)
        if x.min() >= 0 and x.max() <= 1:
            out[c] = "minmax"
        elif np.percentile(x, 99) > 10 * np.median(x):
            out[c] = "robust"
        else:
            out[c] = "standard"
    return out


# ---------------------------------------------------------------- Task 2
def log_transform_skewed(df: pd.DataFrame, threshold: float = 2.0) -> tuple[pd.DataFrame, list[str]]:
    """Apply np.log1p to every numeric column with |skew| > threshold and min >= 0.

    Returns (new_df, transformed_columns). The input DataFrame is NOT modified; the
    returned copy has the same column names (values replaced in place). Non-numeric
    columns and columns with negative values are left alone. Use DataFrame.skew()
    (pandas skips NaN).

    Example: a lognormal column with skew 4.1 comes back as log1p(values) and is
    listed; a uniform column (skew ~0) is untouched and not listed.
    """
    # TODO: copy df, loop over numeric columns, test skew/threshold and min, apply np.log1p, collect names
    out = df.copy()
    changed: list[str] = []
    for c in out.select_dtypes(include="number").columns:
        col = out[c]
        if col.dropna().min() >= 0 and abs(col.skew()) > threshold:
            out[c] = np.log1p(col)
            changed.append(c)
    return out, changed


# ---------------------------------------------------------------- Task 3
def cyclical_encode(df: pd.DataFrame, col: str, period: float) -> pd.DataFrame:
    """Add <col>_sin and <col>_cos = sin/cos(2*pi*value/period) (Chapter 8, Section 5).

    Returns a copy with the two new columns appended (the original column stays).

    Example: hour 7 with period 24 -> hour_sin 0.966, hour_cos -0.259;
             hour 0 and hour 23 land 0.26 apart, hour 0 and hour 12 land 2.0 apart.
    """
    # TODO: angle = 2*pi*df[col]/period; append np.sin(angle) and np.cos(angle) to a copy
    out = df.copy()
    angle = 2 * np.pi * out[col].astype(float) / period
    out[f"{col}_sin"] = np.sin(angle)
    out[f"{col}_cos"] = np.cos(angle)
    return out


# ---------------------------------------------------------------- Task 4
def target_encode_cv(cat_train, y_train, cat_test, n_splits: int = 5,
                     smooth: float = 10.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed target encoding that is cross-fitted on the training rows (Section 3).

    Encoding of category c from a fitting set:  (sum_y_c + smooth * prior) / (n_c + smooth)
    where prior is the mean of y over the fitting set; unseen categories get `prior`.

    * train encoding: KFold(n_splits, shuffle=True, random_state=seed) over the training
      rows; the rows of fold k are encoded with statistics computed from the OTHER folds
      only, so a row's own label never touches its own feature.
    * test encoding: statistics from ALL training rows (new data contributed no labels).

    Returns (train_encoded, test_encoded) as float arrays.

    Example: categories ["a","a","b"] with y [1,1,0], smooth=0 -> test value for "a" is 1.0,
             for "b" 0.0, for "zzz" (unseen) the prior 0.667.
    """
    # TODO: write an inner encode(fit_cat, fit_y, apply_cat) helper, loop over KFold splits for train, use all rows for test
    from sklearn.model_selection import KFold

    cat_train = pd.Series(np.asarray(cat_train)).reset_index(drop=True)
    cat_test = pd.Series(np.asarray(cat_test)).reset_index(drop=True)
    y = pd.Series(np.asarray(y_train, dtype=float)).reset_index(drop=True)

    def encode(fit_cat: pd.Series, fit_y: pd.Series, apply_cat: pd.Series) -> np.ndarray:
        prior = fit_y.mean()
        stats = fit_y.groupby(fit_cat.to_numpy()).agg(["sum", "count"])
        codes = (stats["sum"] + smooth * prior) / (stats["count"] + smooth)
        return apply_cat.map(codes).fillna(prior).to_numpy(dtype=float)

    train_enc = np.empty(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in kf.split(cat_train):
        train_enc[enc_idx] = encode(cat_train.iloc[fit_idx], y.iloc[fit_idx], cat_train.iloc[enc_idx])
    test_enc = encode(cat_train, y, cat_test)
    return train_enc, test_enc


# ---------------------------------------------------------------- Task 5
def build_pipeline(num_cols: list[str], cat_cols: list[str], date_col: str):
    """The ColumnTransformer + Pipeline pattern (Section 9), unfitted.

    Branches:
      "num"  : SimpleImputer(strategy="median", add_indicator=True) -> StandardScaler
      "cat"  : SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore")
      "date" : FunctionTransformer that turns the timestamp column into a DataFrame with
               signup_month and signup_dow (give it feature_names_out=...) -> OneHotEncoder(handle_unknown="ignore")
      "hour" : FunctionTransformer that turns the timestamp into hour_sin, hour_cos (period 24)
    remainder="drop" (the city column and anything else is dropped here — see STRETCH).
    Final step "clf": LogisticRegression(max_iter=1000).

    Returns Pipeline([("pre", ColumnTransformer(...)), ("clf", LogisticRegression(...))]).
    The fitted pipeline must predict_proba on make_new_customers() without error and
    pre.get_feature_names_out() must work (so name every custom transformer's outputs).
    """
    # TODO: define date_parts(X) and hour_parts(X) helpers, wire the four branches into a ColumnTransformer, then the Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    def date_parts(X):
        d = pd.to_datetime(X.iloc[:, 0])
        return pd.DataFrame({"signup_month": d.dt.month.to_numpy(), "signup_dow": d.dt.dayofweek.to_numpy()})

    def hour_parts(X):
        h = pd.to_datetime(X.iloc[:, 0]).dt.hour.to_numpy(dtype=float)
        angle = 2 * np.pi * h / 24
        return np.column_stack([np.sin(angle), np.cos(angle)])

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("date", Pipeline([("parts", FunctionTransformer(date_parts, feature_names_out=lambda tf, names: ["signup_month", "signup_dow"])),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), [date_col]),
        ("hour", FunctionTransformer(hour_parts, feature_names_out=lambda tf, names: ["hour_sin", "hour_cos"]), [date_col]),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


# ---------------------------------------------------------------- Task 6
def text_classifier(docs: list[str], labels: list[int]):
    """Fit make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=10)) (Section 6).

    Returns the fitted pipeline. Its vectorizer step must expose bigrams such as
    "very happy" in get_feature_names_out().

    Example: text_classifier(FEEDBACK, FEEDBACK_LABELS).predict_proba(["very happy, great quality"])[0, 1] > 0.5
    """
    # TODO: build the two-step pipeline and fit it on docs/labels
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=10)).fit(docs, labels)


# ---------------------------------------------------------------- Task 7
def coefficient_table(fitted_pipeline) -> pd.DataFrame:
    """Pair pre.get_feature_names_out() with clf.coef_ (Section 9, "read its output feature names").

    Returns a DataFrame with columns ["feature", "coef"], sorted by |coef| descending,
    index reset. Works for any Pipeline whose steps are named "pre" and "clf" and
    whose classifier is binary (coef_ has shape (1, n_features)).

    Example: the top row for the churn pipeline is a support_calls, plan or
             sign-up-month feature with a positive coefficient.
    """
    # TODO: names = pipeline.named_steps["pre"].get_feature_names_out(); coefs = named_steps["clf"].coef_[0]
    names = fitted_pipeline.named_steps["pre"].get_feature_names_out()
    coefs = fitted_pipeline.named_steps["clf"].coef_[0]
    table = pd.DataFrame({"feature": list(names), "coef": coefs.astype(float)})
    table = table.reindex(table["coef"].abs().sort_values(ascending=False).index)
    return table.reset_index(drop=True)


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    df = make_customer_data()
    print("rows:", len(df), "| churn rate:", round(df.churned.mean(), 3))
    print("scaler per column:", choose_scaler(df, NUM_COLS))
    _, logged = log_transform_skewed(df[NUM_COLS])
    print("log1p applied to:", logged)
    hours = pd.DataFrame({"hour": [0, 7, 14, 21, 23]})
    print(cyclical_encode(hours, "hour", 24).round(3).to_string(index=False))

    X, y = df.drop(columns="churned"), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    tr_enc, te_enc = target_encode_cv(X_tr["city"], y_tr, X_te["city"], seed=0)
    naive = y_tr.groupby(X_tr["city"].to_numpy()).transform("mean").to_numpy()
    print(f"city encoding: AUC of naive (leaky) code on TRAIN rows = {roc_auc_score(y_tr, naive):.3f}; "
          f"cross-fitted = {roc_auc_score(y_tr, tr_enc):.3f}; cross-fitted on TEST rows = {roc_auc_score(y_te, te_enc):.3f}")

    pipe = build_pipeline(NUM_COLS, CAT_COLS, DATE_COL)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"pipeline CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    pipe.fit(X_tr, y_tr)
    print("scores for the five new customers:", pipe.predict_proba(make_new_customers())[:, 1].round(2))
    print(coefficient_table(pipe).head(6).to_string(index=False))

    clf = text_classifier(FEEDBACK, FEEDBACK_LABELS)
    for text in ["still waiting for my refund", "very happy, great quality"]:
        print(f"{text!r:32s} P(praise) = {clf.predict_proba([text])[0, 1]:.2f}")
