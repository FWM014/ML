import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from labs import ch08_feature_pipeline as lab


@pytest.fixture(scope="module")
def data():
    df = lab.make_customer_data(seed=42)
    X, y = df.drop(columns="churned"), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_tr, X_te, y_tr, y_te


def test_task1_scaler_rules_on_controlled_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "share": rng.uniform(0, 1, 500),                      # bounded -> minmax
        "amount": np.append(rng.lognormal(3, 1.5, 499), 1e6),  # heavy tail -> robust
        "age": rng.normal(40, 10, 500),                        # symmetric -> standard
        "with_nan": np.append(rng.normal(5, 1, 499), np.nan),  # NaN must be ignored
    })
    out = lab.choose_scaler(df, ["share", "amount", "age", "with_nan"])
    assert out == {"share": "minmax", "amount": "robust", "age": "standard", "with_nan": "standard"}


def test_task1_scaler_rules_on_customer_table():
    df = lab.make_customer_data()
    out = lab.choose_scaler(df, lab.NUM_COLS)
    assert out["discount_share"] == "minmax"
    assert out["data_gb"] == "robust"
    assert out["tenure_months"] == "standard"


def test_task2_log_transform_only_skewed_nonneg_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "skewed": rng.lognormal(4, 1.0, 2000),
        "flat": rng.uniform(0, 10, 2000),
        "signed": rng.lognormal(4, 1.0, 2000) - 500.0,  # skewed but negative -> untouched
        "label": rng.choice(["a", "b"], 2000),
    })
    original = df.copy()
    out, changed = lab.log_transform_skewed(df, threshold=2.0)
    assert changed == ["skewed"]
    assert list(out.columns) == list(df.columns)
    assert np.allclose(out["skewed"], np.log1p(df["skewed"]))
    assert np.allclose(out["flat"], df["flat"]) and np.allclose(out["signed"], df["signed"])
    assert abs(out["skewed"].skew()) < 0.5  # tail is tamed
    pd.testing.assert_frame_equal(df, original)  # input not modified


def test_task3_cyclical_encoding_matches_chapter_and_wraps_around():
    df = pd.DataFrame({"hour": [0, 7, 14, 21, 23, 12]})
    out = lab.cyclical_encode(df, "hour", 24)
    assert {"hour", "hour_sin", "hour_cos"} <= set(out.columns)
    assert out.loc[1, "hour_sin"] == pytest.approx(0.966, abs=1e-3)
    assert out.loc[1, "hour_cos"] == pytest.approx(-0.259, abs=1e-3)
    pts = out[["hour_sin", "hour_cos"]].to_numpy()
    d = lambda i, j: np.linalg.norm(pts[i] - pts[j])
    assert d(0, 4) < 0.3          # 00:00 is next to 23:00
    assert d(0, 5) == pytest.approx(2.0, abs=1e-6)  # 00:00 is opposite 12:00
    assert np.allclose(pts[:, 0] ** 2 + pts[:, 1] ** 2, 1.0)
    assert "hour_sin" not in df.columns  # input not modified


def test_task4_target_encoding_values_and_unseen_categories():
    tr_enc, te_enc = lab.target_encode_cv(["a", "a", "b"], [1, 1, 0], ["a", "b", "zzz"], n_splits=3, smooth=0.0)
    assert te_enc.tolist() == pytest.approx([1.0, 0.0, 2 / 3])
    # smoothing pulls a rare category toward the prior
    _, te_s = lab.target_encode_cv(["a", "a", "b"], [1, 1, 0], ["b"], n_splits=3, smooth=10.0)
    assert 0.0 < te_s[0] < 2 / 3
    assert tr_enc.shape == (3,) and te_enc.dtype.kind == "f"


def test_task4_cross_fitting_does_not_leak_the_label():
    # a pure-noise id column with ~2 rows per level: naive encoding is nearly the label itself
    rng = np.random.default_rng(7)
    n = 3000
    ids = rng.integers(0, 1500, n).astype(str)
    y = rng.integers(0, 2, n)
    naive = pd.Series(y, dtype=float).groupby(ids).transform("mean").to_numpy()
    assert roc_auc_score(y, naive) > 0.80  # the leak, for reference
    tr_enc, _ = lab.target_encode_cv(ids, y, ids, n_splits=5, smooth=10.0, seed=0)
    assert roc_auc_score(y, tr_enc) < 0.58  # honest: no information in the ids
    # a category that occurs once must be encoded with NO trace of its own label
    counts = pd.Series(ids).value_counts()
    single = np.isin(ids, counts[counts == 1].index)
    assert single.sum() > 100
    assert abs(np.corrcoef(tr_enc[single], y[single])[0, 1]) < 0.1
    # reproducible
    tr_enc2, _ = lab.target_encode_cv(ids, y, ids, n_splits=5, smooth=10.0, seed=0)
    assert np.allclose(tr_enc, tr_enc2)


def test_task4_city_carries_real_signal_when_encoded_honestly(data):
    X_tr, X_te, y_tr, y_te = data
    tr_enc, te_enc = lab.target_encode_cv(X_tr["city"], y_tr, X_te["city"], seed=0)
    assert roc_auc_score(y_te, te_enc) > 0.55  # generalises to unseen rows


def test_task5_pipeline_survives_nan_and_unknown_categories(data):
    X_tr, X_te, y_tr, y_te = data
    pipe = lab.build_pipeline(lab.NUM_COLS, lab.CAT_COLS, lab.DATE_COL)
    assert list(pipe.named_steps) == ["pre", "clf"]
    pipe.fit(X_tr, y_tr)
    new = lab.make_new_customers()
    p = pipe.predict_proba(new)[:, 1]
    assert p.shape == (5,) and np.all((p >= 0) & (p <= 1))
    # the 3-month basic customer with 4 calls is riskier than the 60-month premium one
    assert p[0] > p[3]
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    assert any("missingindicator" in n for n in names)
    assert any(n.endswith("plan_basic") for n in names)
    assert any("hour_sin" in n for n in names) and any("signup_month" in n for n in names)
    assert not any("city" in n for n in names)


def test_task5_pipeline_scores_honestly_in_cv(data):
    X_tr, X_te, y_tr, y_te = data
    pipe = lab.build_pipeline(lab.NUM_COLS, lab.CAT_COLS, lab.DATE_COL)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    assert scores.mean() > 0.65
    pipe.fit(X_tr, y_tr)
    assert roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]) > 0.65


def test_task6_text_classifier_reads_bigrams():
    clf = lab.text_classifier(lab.FEEDBACK, lab.FEEDBACK_LABELS)
    vocab = set(clf[0].get_feature_names_out())
    assert "very happy" in vocab and "refund" in vocab
    assert clf.predict_proba(["still waiting for my refund"])[0, 1] < 0.4
    assert clf.predict_proba(["very happy, great quality"])[0, 1] > 0.6
    assert clf.predict(["broken and damaged, want a refund", "excellent, love it"]).tolist() == [0, 1]


def test_task7_coefficient_table_pairs_names_with_coefs(data):
    X_tr, X_te, y_tr, y_te = data
    pipe = lab.build_pipeline(lab.NUM_COLS, lab.CAT_COLS, lab.DATE_COL).fit(X_tr, y_tr)
    tbl = lab.coefficient_table(pipe)
    assert list(tbl.columns) == ["feature", "coef"]
    assert len(tbl) == pipe.named_steps["clf"].coef_.shape[1]
    assert tbl["coef"].abs().is_monotonic_decreasing
    assert list(tbl.index) == list(range(len(tbl)))
    coef = tbl.set_index("feature")["coef"]
    assert coef["num__support_calls"] > 0.3
    assert coef["num__tenure_months"] < -0.3
    assert coef["cat__plan_basic"] > coef["cat__plan_premium"]
