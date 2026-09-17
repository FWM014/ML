import os

import numpy as np
import pandas as pd
import pytest

from labs import ch04_pandas_sklearn as lab


@pytest.fixture(scope="module")
def csv_path():
    return lab.make_orders_csv(seed=0)


@pytest.fixture(scope="module")
def raw_frame(csv_path):
    # what the CSV really contains, read naively (amount is text because of "pending")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def orders(csv_path):
    return lab.load_orders(csv_path)


def test_task1_load_orders_dtypes_and_coercion(orders, raw_frame):
    assert raw_frame["amount"].dtype.kind in "OU"           # naive read: text
    assert orders["ordered"].dtype.kind == "M"
    assert orders["amount"].dtype == np.float64
    assert orders["amount"].isna().sum() == (raw_frame["amount"] == "pending").sum() == 6
    assert len(orders) == len(raw_frame) == 400
    assert list(orders.columns) == ["order_id", "customer_id", "region", "amount", "ordered"]


def test_task2_revenue_by_region(orders):
    out = lab.revenue_by_region(orders)
    assert list(out.columns) == ["n_orders", "revenue", "avg_amount"]
    assert set(out.index) == {"north", "south", "east", "west"}
    assert out["revenue"].is_monotonic_decreasing
    assert out.index[0] == "north"
    north = orders[orders["region"] == "north"]
    assert out.loc["north", "n_orders"] == len(north)               # NaN rows still count as orders
    assert out.loc["north", "revenue"] == pytest.approx(north["amount"].sum())
    assert out.loc["north", "avg_amount"] == pytest.approx(north["amount"].mean())
    assert out["n_orders"].sum() == 400


def test_task3_attach_customer_tier_keeps_every_order(orders):
    customers = lab.make_customers()
    joined = lab.attach_customer_tier(orders, customers)
    assert len(joined) == len(orders)                                  # left join: no rows lost
    assert "tier" in joined.columns and "tier" not in orders.columns  # input untouched
    missing = ~orders["customer_id"].isin(customers["customer_id"])
    assert missing.sum() > 0
    assert (joined.loc[missing.values, "tier"] == "unknown").all()
    assert joined["tier"].isna().sum() == 0
    lookup = customers.set_index("customer_id")["tier"]
    matched = joined[~missing.values]
    assert (matched["tier"].values == lookup.loc[matched["customer_id"]].values).all()


def test_task4_add_time_features():
    df = pd.DataFrame({"ordered": pd.to_datetime(["2024-01-01", "2024-01-06", "2024-03-15"])})
    out = lab.add_time_features(df)
    assert out["month"].tolist() == [1, 1, 3]
    assert out["weekday"].tolist() == [0, 5, 4]
    assert out["is_weekend"].tolist() == [False, True, False]
    assert out["days_since_first"].tolist() == [0, 5, 74]
    assert "month" not in df.columns                                   # returned a new frame


def test_task5_pipeline_trained_on_training_rows_only():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    res = lab.train_tumour_model(seed=42)
    model = res["model"]
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scale"], StandardScaler)
    assert res["test_accuracy"] > 0.95
    cm = np.asarray(res["confusion_matrix"])
    assert cm.shape == (2, 2) and cm.sum() == 114
    assert len(res["feature_names"]) == 30
    # the classic mistake: scaler statistics from ALL rows would leak the test set
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    assert np.allclose(model.named_steps["scale"].mean_, X_tr.mean(axis=0))
    assert not np.allclose(model.named_steps["scale"].mean_, X.mean(axis=0))


def test_task6_save_and_reload_round_trip(tmp_path):
    from sklearn.datasets import load_breast_cancer

    res = lab.train_tumour_model(seed=1)
    path = tmp_path / "models" / "tumour.joblib"                       # parent dir does not exist yet
    loaded = lab.save_and_reload(res["model"], str(path))
    assert path.exists()
    X, _ = load_breast_cancer(return_X_y=True)
    assert np.array_equal(loaded.predict(X), res["model"].predict(X))
    assert np.allclose(loaded.predict_proba(X[:1]), res["model"].predict_proba(X[:1]))
    assert loaded is not res["model"]


def test_task7_top_k_coefficients():
    class FakeClf:
        coef_ = np.array([[0.5, -2.0, 1.0, 0.1]])

    class FakePipe:
        named_steps = {"clf": FakeClf()}

    out = lab.top_k_coefficients(FakePipe(), ["a", "b", "c", "d"], k=2)
    assert list(out.columns) == ["feature", "coef"]
    assert out["feature"].tolist() == ["b", "c"]
    assert out["coef"].tolist() == [-2.0, 1.0]
    assert out.index.tolist() == [0, 1]
    res = lab.train_tumour_model(seed=42)
    top = lab.top_k_coefficients(res["model"], res["feature_names"], k=5)
    assert len(top) == 5
    assert top["coef"].abs().is_monotonic_decreasing
    assert set(top["feature"]) <= set(res["feature_names"])
    coef = res["model"].named_steps["clf"].coef_.ravel()
    assert top["coef"].abs().iloc[0] == pytest.approx(np.abs(coef).max())
