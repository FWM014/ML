import numpy as np
import pandas as pd
import pytest

from labs import ch02_data_audit as lab


@pytest.fixture(scope="module")
def raw():
    return lab.make_messy_customers()


@pytest.fixture(scope="module")
def clean(raw):
    return lab.normalise_and_dedupe(lab.fix_sentinels(raw, "age", -999), ["plan"])


def test_task1_class_balance_small_example():
    res = lab.class_balance(np.array([0, 0, 0, 1]))
    assert res["counts"] == {0: 3, 1: 1}
    assert res["fractions"][0] == pytest.approx(0.75)
    assert res["fractions"][1] == pytest.approx(0.25)
    assert res["majority_baseline"] == pytest.approx(0.75)


def test_task1_class_balance_on_export(raw):
    res = lab.class_balance(raw["churned"])
    assert res["counts"][0] + res["counts"][1] == len(raw)
    assert 0.20 < res["fractions"][1] < 0.32
    assert res["majority_baseline"] == pytest.approx(max(res["fractions"].values()))


def test_task2_missingness_report_shape_and_order(raw):
    rep = lab.missingness_report(raw)
    assert list(rep.columns) == ["n_missing", "frac_missing"]
    assert set(rep.index) == set(raw.columns)
    assert rep.index[0] == "cancelled_date"           # the leaky column is mostly empty
    assert rep.loc["monthly_spend", "n_missing"] == raw["monthly_spend"].isna().sum()
    assert rep.loc["age", "n_missing"] == 0             # the sentinel hides as a real number
    assert rep["frac_missing"].is_monotonic_decreasing
    assert rep["frac_missing"].max() == pytest.approx(rep["n_missing"].max() / len(raw))


def test_task3_fix_sentinels_transforms_without_deleting(raw):
    fixed = lab.fix_sentinels(raw, "age", -999)
    n_sentinel = int((raw["age"] == -999).sum())
    assert n_sentinel > 0                                # the export really has them
    assert raw["age"].min() == -999                      # input was NOT modified
    assert len(fixed) == len(raw)                        # nothing deleted
    assert fixed["age"].isna().sum() == n_sentinel
    assert fixed["age"].min() >= 18
    assert fixed["age_missing"].sum() == n_sentinel
    assert set(fixed["age_missing"].unique()) <= {0, 1}
    # the sentinel had wrecked the statistic; now it is plausible
    assert 40 < fixed["age"].mean() < 50


def test_task4_normalise_and_dedupe(raw):
    assert raw["plan"].nunique() > 3
    assert raw.duplicated().sum() > 0
    out = lab.normalise_and_dedupe(raw, ["plan"])
    assert sorted(out["plan"].unique()) == ["basic", "plus", "premium"]
    assert out.duplicated().sum() == 0
    assert len(out) == 1200
    assert out.index.tolist() == list(range(len(out)))
    assert raw.duplicated().sum() > 0                    # input untouched


def test_task5_find_suspect_columns_by_rule(clean):
    res = lab.find_suspect_columns(clean, "churned")
    assert res["id"] == ["customer_id"]
    assert res["leaky"] == ["cancelled_date"]
    honest = {"age", "tenure_months", "monthly_spend", "support_calls", "days_since_login",
              "autopay", "plan", "signup_date", "age_missing"}
    assert honest.isdisjoint(res["id"]) and honest.isdisjoint(res["leaky"])


def test_task5_numeric_leak_and_target_excluded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame({
        "row_id": np.arange(300),
        "x": rng.normal(size=300),
        "refund_issued": y * (rng.random(300) < 0.98),   # caused by the label
        "label": y,
    })
    res = lab.find_suspect_columns(df, "label")
    assert res["id"] == ["row_id"]
    assert res["leaky"] == ["refund_issued"]


def test_task6_time_split_never_shuffles(clean):
    train, test = lab.time_split(clean, "signup_date", test_frac=0.2)
    assert len(train) + len(test) == len(clean)
    assert abs(len(test) - 0.2 * len(clean)) < 0.03 * len(clean)
    assert train["signup_date"].max() < test["signup_date"].min()
    assert set(train.index).isdisjoint(test.index)


def test_task7_stratified_split_sizes_balance_and_no_overlap(clean):
    train, val, test = lab.stratified_split(clean, "churned", fracs=(0.6, 0.2, 0.2), seed=0)
    n = len(clean)
    assert abs(len(train) - 0.6 * n) <= 2 and abs(len(val) - 0.2 * n) <= 2 and abs(len(test) - 0.2 * n) <= 2
    idx = [set(p.index) for p in (train, val, test)]
    assert idx[0].isdisjoint(idx[1]) and idx[0].isdisjoint(idx[2]) and idx[1].isdisjoint(idx[2])
    assert idx[0] | idx[1] | idx[2] == set(clean.index)
    rate = clean["churned"].mean()
    for part in (train, val, test):
        assert abs(part["churned"].mean() - rate) < 0.015
    # seeded -> reproducible
    train2, _, _ = lab.stratified_split(clean, "churned", fracs=(0.6, 0.2, 0.2), seed=0)
    assert train2.index.tolist() == train.index.tolist()
