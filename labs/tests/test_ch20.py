import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from labs import ch20_interpretability as lab


@pytest.fixture(scope="module")
def data():
    X, y, group = lab.make_loan_data(seed=0)
    X_tr, X_va, y_tr, y_va, g_tr, g_va = train_test_split(X, y, group, test_size=0.3, random_state=0, stratify=y)
    gbm = lab.fit_gbm(X_tr, y_tr)
    return dict(X_tr=X_tr, X_va=X_va, y_tr=y_tr, y_va=y_va, g_tr=g_tr, g_va=g_va, gbm=gbm)


# ---------------------------------------------------------------- Task 1
def test_task1_permutation_importance_matches_sklearn(data):
    gbm, X_va, y_va = data["gbm"], data["X_va"], data["y_va"]
    mine = lab.permutation_importance_manual(gbm, X_va, y_va, n_repeats=15, seed=0)
    assert isinstance(mine, pd.Series) and set(mine.index) == set(X_va.columns)
    assert list(mine.values) == sorted(mine.values, reverse=True), "must be sorted descending"
    sk = permutation_importance(gbm, X_va, y_va, scoring="neg_log_loss", n_repeats=15, random_state=0)
    sk_mean = pd.Series(sk.importances_mean, index=X_va.columns)
    sk_std = pd.Series(sk.importances_std, index=X_va.columns)
    for f in X_va.columns:
        tol = 0.02 + 3 * sk_std[f] / np.sqrt(15)
        assert abs(mine[f] - sk_mean[f]) < tol, f"{f}: {mine[f]:.4f} vs sklearn {sk_mean[f]:.4f}"
    # the two strongest drivers are the ones the data generator used most heavily
    assert set(mine.index[:2]) == {"credit_score", "income"}
    # a feature the outcome never depended on is ~0 (the classic sanity check on importances)
    assert abs(mine["years_employed"]) < 0.02


def test_task1_is_reproducible(data):
    a = lab.permutation_importance_manual(data["gbm"], data["X_va"], data["y_va"], n_repeats=5, seed=3)
    b = lab.permutation_importance_manual(data["gbm"], data["X_va"], data["y_va"], n_repeats=5, seed=3)
    assert np.allclose(a.sort_index().values, b.sort_index().values)


# ---------------------------------------------------------------- Task 2
def test_task2_partial_dependence_matches_sklearn(data):
    gbm, X_va = data["gbm"], data["X_va"]
    for feat in ["income", "credit_score"]:
        res = partial_dependence(gbm, X_va, features=[feat], kind="average", grid_resolution=8,
                                 method="brute", response_method="predict_proba")
        grid = np.asarray(res["grid_values"][0])
        mine = lab.partial_dependence_manual(gbm, X_va, feat, grid)
        assert isinstance(mine, np.ndarray) and mine.shape == (len(grid),)
        assert np.allclose(mine, res["average"][0], atol=1e-6), feat
    # PDP of P(repaid) should rise with income (that is how the data were generated)
    grid = np.linspace(20, 80, 5)
    curve = lab.partial_dependence_manual(gbm, X_va, "income", grid)
    assert curve[-1] > curve[0] + 0.2
    assert np.all((curve >= 0) & (curve <= 1))


# ---------------------------------------------------------------- Task 3
def test_task3_shapley_taxi_game():
    v = {(): 0, ("A",): 6, ("B",): 12, ("C",): 42, ("A", "B"): 12, ("A", "C"): 42, ("B", "C"): 42, ("A", "B", "C"): 42}
    phi = lab.shapley_values(v, ["A", "B", "C"])
    assert phi["A"] == pytest.approx(2.0) and phi["B"] == pytest.approx(5.0) and phi["C"] == pytest.approx(35.0)


def test_task3_shapley_hand_computed_and_efficiency():
    # hand-computed: A = 10/3 + (40+40)/6 + 30/3 = 26.667, B = 41.667, C = 51.667 (sum = 120)
    v = {(): 0, ("A",): 10, ("B",): 20, ("C",): 30, ("A", "B"): 60, ("A", "C"): 70, ("B", "C"): 90, ("A", "B", "C"): 120}
    phi = lab.shapley_values(v, ["A", "B", "C"])
    assert phi["A"] == pytest.approx(80 / 3)
    assert phi["B"] == pytest.approx(125 / 3)
    assert phi["C"] == pytest.approx(155 / 3)
    assert sum(phi.values()) == pytest.approx(120)
    # symmetry: interchangeable players get equal credit; a null player gets 0
    v2 = {(): 0, ("x",): 1, ("y",): 1, ("z",): 0, ("x", "y"): 4, ("x", "z"): 1, ("y", "z"): 1, ("x", "y", "z"): 4}
    phi2 = lab.shapley_values(v2, ["x", "y", "z"])
    assert phi2["x"] == pytest.approx(phi2["y"]) and phi2["z"] == pytest.approx(0.0)


# ---------------------------------------------------------------- Task 4
def test_task4_tree_shap_efficiency(data):
    rf = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=0).fit(data["X_tr"], data["y_tr"])
    rows = data["X_va"].head(20)
    out = lab.tree_shap_explain(rf, rows)
    assert set(out) >= {"base", "shap", "prediction", "efficiency_gap", "global_importance"}
    assert np.asarray(out["shap"]).shape == (20, rows.shape[1])
    assert 0.0 < out["base"] < 1.0
    recon = out["base"] + np.asarray(out["shap"]).sum(axis=1)
    assert np.allclose(recon, rf.predict_proba(rows)[:, 1], atol=1e-6)
    assert out["efficiency_gap"] < 1e-6
    gi = out["global_importance"]
    assert isinstance(gi, pd.Series) and list(gi.values) == sorted(gi.values, reverse=True)
    assert gi.index[0] in {"credit_score", "income"}


# ---------------------------------------------------------------- Task 5
def test_task5_counterfactual_flips_and_is_minimal(data):
    X_tr, y_tr, X_va = data["X_tr"], data["y_tr"], data["X_va"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    declined = X_va[lr.predict(X_va) == 0]
    n_checked = 0
    for i in range(3):
        row = declined.iloc[i]
        cf = lab.counterfactual(lr, row, X_tr, target_class=1, max_std=3.0, n_steps=61)
        if cf is None:
            continue
        n_checked += 1
        assert set(cf) >= {"feature", "old", "new", "step_std"}
        # only ONE feature changed, and the new row really is approved
        x = pd.DataFrame([row]); x[cf["feature"]] = cf["new"]
        assert int(lr.predict(x)[0]) == 1
        assert cf["old"] == pytest.approx(float(row[cf["feature"]]))
        assert cf["new"] == pytest.approx(cf["old"] + cf["step_std"] * X_tr[cf["feature"]].std())
        # analytic check for a linear model: the true minimal move is -logit / (w_f * sd_f/scale_f)
        sc, clf = lr[0], lr[-1]
        logit = clf.decision_function(sc.transform(pd.DataFrame([row])))[0]
        need = {}
        for j, f in enumerate(X_tr.columns):
            w_std = clf.coef_[0][j] * X_tr[f].std() / sc.scale_[j]      # logit change per 1 std of f
            if abs(w_std) > 1e-9:
                need[f] = -logit / w_std
        feasible = {f: s for f, s in need.items() if abs(s) <= 3.0}
        best_f = min(feasible, key=lambda f: abs(feasible[f]))
        assert abs(cf["step_std"]) <= abs(feasible[best_f]) + 0.1 + 1e-9, "not the smallest move"
    assert n_checked >= 2


def test_task5_returns_none_when_nothing_flips():
    X = pd.DataFrame({"a": np.linspace(-1, 1, 50), "b": np.linspace(-1, 1, 50)[::-1]})
    y = np.r_[np.zeros(25), np.ones(25)]
    lr = LogisticRegression().fit(X, y)
    row = pd.Series({"a": -50.0, "b": 50.0})
    assert lab.counterfactual(lr, row, X, target_class=1, max_std=0.5, n_steps=11) is None


# ---------------------------------------------------------------- Task 6
def test_task6_fairness_hand_computed():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = lab.fairness_metrics(y_true, y_pred, group)
    assert m["selection_rate"] == pytest.approx({0: 0.75, 1: 0.25})
    assert m["tpr"] == pytest.approx({0: 1.0, 1: 0.5})
    assert m["demographic_parity_diff"] == pytest.approx(0.5)
    assert m["equal_opportunity_diff"] == pytest.approx(0.5)
    assert m["disparate_impact_ratio"] == pytest.approx(1 / 3)


def test_task6_proxy_feature_creates_a_gap(data):
    X_tr, y_tr, X_va, y_va, g_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], data["g_va"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    m = lab.fairness_metrics(y_va, lr.predict(X_va), g_va)
    assert m["disparate_impact_ratio"] < 0.8          # fails the 4/5 rule although group is not a feature
    assert m["equal_opportunity_diff"] > 0.1
    # a perfectly even decision has zero gaps
    even = lab.fairness_metrics(y_va, np.ones_like(y_va), g_va)
    assert even["demographic_parity_diff"] == pytest.approx(0.0)
    assert even["disparate_impact_ratio"] == pytest.approx(1.0)
