import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from labs import ch21_mlops as lab


@pytest.fixture(scope="module")
def churn():
    df = lab.make_churn_data(seed=0)
    pipe = lab.fit_churn_pipeline(df)
    return df, pipe


@pytest.fixture
def scratch():
    """Files go under labs/data/ (the only place labs may write) and are removed afterwards."""
    os.makedirs(lab.DATA_DIR, exist_ok=True)
    paths = []
    yield lambda name: paths.append(os.path.join(lab.DATA_DIR, name)) or paths[-1]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------- Task 1
def test_task1_log_run_appends_and_best_run_queries(scratch):
    path = scratch("ch21_test_log.jsonl")
    r1 = lab.log_run(path, "logreg", {"C": 1.0}, {"cv_auc": 0.81, "fit_s": 0.2}, tags=("a",))
    r2 = lab.log_run(path, "rf", {"n_estimators": 200}, {"cv_auc": 0.86})
    r3 = lab.log_run(path, "gbm", {"depth": 3}, {"cv_auc": 0.84, "fit_s": 2.0})
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(lines) == 3, "append-only: one JSON line per run"
    assert [l["run_id"] for l in lines] == [r1, r2, r3]
    assert len(set([r1, r2, r3])) == 3 and all(len(r) == 10 for r in (r1, r2, r3))
    assert lines[0]["sklearn"] == sklearn.__version__ and lines[0]["tags"] == ["a"] and lines[0]["params"] == {"C": 1.0}
    best = lab.best_run(path, "cv_auc")
    assert best["name"] == "rf" and best["run_id"] == r2
    assert lab.best_run(path, "fit_s", higher_is_better=False)["name"] == "logreg"
    with pytest.raises(ValueError):
        lab.best_run(path, "does_not_exist")


# ---------------------------------------------------------------- Task 2
def test_task2_bundle_roundtrip_and_version_guard(churn, scratch):
    df, pipe = churn
    path = scratch("ch21_test_bundle.joblib")
    assert lab.save_bundle(pipe, path, "churn-v9.9.9", extra={"note": "test"}) == path
    b = lab.load_bundle(path)
    assert b["model_version"] == "churn-v9.9.9" and b["sklearn_version"] == sklearn.__version__
    assert b["features"] == lab.FEATURES and b["note"] == "test"
    X = df[lab.FEATURES].head(30)
    assert np.allclose(b["pipeline"].predict_proba(X), pipe.predict_proba(X))
    # a bundle pickled under another sklearn: strict refuses, lenient warns
    old = scratch("ch21_test_bundle_old.joblib")
    joblib.dump(dict(b, sklearn_version="1.2.2"), old)
    with pytest.raises(RuntimeError):
        lab.load_bundle(old)
    lenient = lab.load_bundle(old, strict=False)
    assert "warning" in lenient and "1.2.2" in lenient["warning"]


# ---------------------------------------------------------------- Task 3
def test_task3_psi_hand_computed():
    ref = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    cur = np.array([1, 1, 1, 1, 2, 2, 3, 4])
    assert lab.psi(ref, cur, bins=[0.5, 1.5, 2.5, 3.5, 4.5]) == pytest.approx(0.5 * np.log(2), abs=1e-3)
    assert lab.psi(ref, ref, bins=[0.5, 1.5, 2.5, 3.5, 4.5]) == pytest.approx(0.0, abs=1e-4)
    rng = np.random.default_rng(0)
    a = rng.normal(100, 15, 20000)
    assert lab.psi(a, rng.normal(100, 15, 5000)) < 0.02          # same distribution
    assert lab.psi(a, rng.normal(112, 15, 5000)) > 0.4           # big shift
    # values outside the reference range must still be counted (outer edges are open)
    assert lab.psi(a, np.full(5000, 1000.0)) > 2.0


def test_task3_drift_alert_verdicts():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000), "c": rng.normal(0, 1, 5000),
                        "plan": rng.choice(["x", "y"], 5000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0.4, 1, 2000), "c": rng.normal(1.2, 1, 2000),
                        "plan": rng.choice(["x", "y"], 2000)})
    rep = lab.drift_alert(ref, cur)
    assert list(rep.index) == ["a", "b", "c"] and set(rep.columns) >= {"psi", "verdict"}
    assert rep.loc["a", "verdict"] == "stable"
    assert rep.loc["b", "verdict"] == "investigate"
    assert rep.loc["c", "verdict"] == "alert"


# ---------------------------------------------------------------- Task 4
def test_task4_validate_batch_both_paths(churn):
    df, _ = churn
    good = df[lab.FEATURES].head(100).copy()
    good.loc[good.index[3], "support_calls"] = np.nan             # one null in a nullable column is fine
    assert lab.validate_batch(good) is True
    bad = good.copy()
    bad.loc[bad.index[0], "age"] = 340                             # two of 100 rows = 2% out of range > 1%
    bad.loc[bad.index[1], "age"] = -5
    bad.loc[bad.index[5], "plan"] = "gold"
    bad = bad.drop(columns="tenure_months")
    bad["extra_col"] = 1
    with pytest.raises(lab.DataValidationError) as e:
        lab.validate_batch(bad)
    msg = str(e.value)
    assert "tenure_months" in msg and "extra_col" in msg and "age" in msg and "gold" in msg
    # a null in a NON-nullable column is rejected even if it is a single row
    bad2 = good.copy(); bad2.loc[bad2.index[0], "age"] = np.nan
    with pytest.raises(lab.DataValidationError):
        lab.validate_batch(bad2)
    # wrong dtype is rejected
    bad3 = good.copy(); bad3["monthly_spend"] = bad3["monthly_spend"].astype(str)
    with pytest.raises(lab.DataValidationError):
        lab.validate_batch(bad3)


# ---------------------------------------------------------------- Task 5
def test_task5_detect_skew_finds_the_wrong_feature_code(churn):
    df, _ = churn
    train = df[lab.FEATURES].head(2000)
    serving = df[lab.FEATURES].tail(800).copy()
    serving["tenure_months"] = serving["tenure_months"] / 12        # serving code computed YEARS
    serving = serving.drop(columns="support_calls")
    serving["request_id"] = np.arange(len(serving))
    out = lab.detect_skew(train, serving)
    assert out["missing_in_serving"] == ["support_calls"]
    assert out["extra_in_serving"] == ["request_id"]
    assert out["skewed"] == ["tenure_months"]
    rep = out["report"]
    assert set(rep.columns) >= {"train_mean", "serve_mean", "psi", "ks_p", "skewed"}
    assert set(rep.index) == {"age", "monthly_spend", "tenure_months"}
    assert rep.loc["tenure_months", "psi"] > 0.5 and rep.loc["age", "psi"] < 0.1
    # identical feature code -> nothing flagged
    clean = lab.detect_skew(train, df[lab.FEATURES].tail(800))
    assert clean["skewed"] == [] and clean["missing_in_serving"] == []


# ---------------------------------------------------------------- Task 6
def test_task6_backfill_respects_label_delay():
    log = lab.make_prediction_log(seed=7)
    d44 = lab.backfill_performance(log, today=44)
    assert int(d44["pred_rate"].index.max()) == 44                  # prediction rate is instant
    assert d44["latest_labelled_day"] == 44 - 21                    # accuracy lags by the label delay
    assert int(d44["accuracy"].index.max()) == 23
    assert 0.6 < d44["accuracy"].mean() < 0.85
    d89 = lab.backfill_performance(log, today=89)
    early = d89["accuracy"].loc[:44].mean()
    late = d89["accuracy"].loc[45:].mean()
    assert early - late > 0.15, "concept drift after day 45 must show up in backfilled accuracy"
    # ... while the prediction rate is blind to it
    assert abs(d89["pred_rate"].loc[:44].mean() - d89["pred_rate"].loc[45:].mean()) < 0.03
    # before any label has arrived there is no accuracy at all
    d5 = lab.backfill_performance(log, today=5)
    assert d5["latest_labelled_day"] == -1 and len(d5["accuracy"]) == 0
    # the future must not leak into today's view
    assert d5["pred_rate"].index.max() == 5


# ---------------------------------------------------------------- Task 7
def test_task7_predict_payload_valid(churn):
    df, pipe = churn
    bundle = {"pipeline": pipe, "model_version": "churn-v1.0.0", "features": lab.FEATURES,
              "sklearn_version": sklearn.__version__}
    out = lab.predict_payload(bundle, {"customer_id": "c-17", "age": 40, "monthly_spend": 30.5, "plan": "basic",
                                       "support_calls": 4, "tenure_months": 3})
    assert out["ok"] is True and out["customer_id"] == "c-17" and out["model_version"] == "churn-v1.0.0"
    assert 0.0 <= out["churn_probability"] <= 1.0
    X = pd.DataFrame([{"age": 40.0, "monthly_spend": 30.5, "plan": "basic", "support_calls": 4.0, "tenure_months": 3.0}])
    assert out["churn_probability"] == pytest.approx(float(pipe.predict_proba(X[lab.FEATURES])[0, 1]))
    # a nullable feature may be absent; key order in the payload must not matter
    out2 = lab.predict_payload(bundle, {"tenure_months": 3, "plan": "basic", "monthly_spend": 30.5, "age": 40})
    assert out2["ok"] is True and out2["customer_id"] is None
    # more support calls -> higher churn risk (directional expectation)
    hi = lab.predict_payload(bundle, {"age": 40, "monthly_spend": 30.5, "plan": "basic", "support_calls": 9, "tenure_months": 3})
    assert hi["churn_probability"] > out["churn_probability"]


def test_task7_predict_payload_rejects_bad_input(churn):
    df, pipe = churn
    bundle = {"pipeline": pipe, "model_version": "churn-v1.0.0", "features": lab.FEATURES,
              "sklearn_version": sklearn.__version__}
    bad = lab.predict_payload(bundle, {"customer_id": "c-1", "age": 340, "monthly_spend": "lots", "plan": "gold"})
    assert bad["ok"] is False and bad["customer_id"] == "c-1"
    joined = " | ".join(bad["errors"])
    assert "age" in joined and "monthly_spend" in joined and "plan" in joined and "tenure_months" in joined
    assert "churn_probability" not in bad
    # booleans are not numbers; never raise
    weird = lab.predict_payload(bundle, {"age": True, "monthly_spend": 10, "plan": "plus", "tenure_months": 1})
    assert weird["ok"] is False and any("age" in e for e in weird["errors"])
    assert lab.predict_payload(bundle, {})["ok"] is False
