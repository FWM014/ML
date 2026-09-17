import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from labs import ch09_decision_trees as lab


@pytest.fixture(scope="module")
def loans():
    X, y = lab.make_collections_data(seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te


def test_task1_gini_and_entropy_by_hand():
    assert lab.impurity([5, 5]) == pytest.approx(0.5)
    assert lab.impurity([5, 5], "entropy") == pytest.approx(1.0)
    assert lab.impurity([8, 2]) == pytest.approx(0.32)
    assert lab.impurity([8, 2], "entropy") == pytest.approx(0.7219, abs=1e-3)
    assert lab.impurity([4, 0]) == pytest.approx(0.0)
    assert lab.impurity([4, 0], "entropy") == pytest.approx(0.0)  # no log(0) blow-up
    assert lab.impurity([1, 1, 1], "entropy") == pytest.approx(np.log2(3))
    with pytest.raises(ValueError):
        lab.impurity([1, 1], "chi2")


def test_task2_split_table_matches_the_worked_example():
    tbl = lab.split_candidates(lab.CALLS, lab.CHURNED)
    assert list(tbl.columns) == ["threshold", "weighted_gini", "info_gain"]
    assert tbl["threshold"].tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    assert tbl["weighted_gini"].tolist() == pytest.approx([0.400, 0.267, 0.317, 0.419, 0.475, 0.400], abs=1e-3)
    # information gain with the true 6/4 parent (entropy 0.971)
    assert tbl["info_gain"].tolist() == pytest.approx([0.171, 0.420, 0.256, 0.091, 0.007, 0.144], abs=1e-3)
    assert tbl.loc[tbl["weighted_gini"].idxmin(), "threshold"] == 1.5
    assert tbl.loc[tbl["info_gain"].idxmax(), "threshold"] == 1.5


def test_task3_best_split_matches_sklearn_root(loans):
    X, y, *_ = loans
    j, thr, g = lab.best_split(X, y)
    sk = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    t = sk.tree_
    assert j == t.feature[0]
    assert thr == pytest.approx(t.threshold[0], abs=1e-6)
    n = t.n_node_samples
    sk_weighted = (n[1] * t.impurity[1] + n[2] * t.impurity[2]) / n[0]
    assert g == pytest.approx(sk_weighted, abs=1e-6)
    assert g < t.impurity[0]  # the split actually reduces impurity


def test_task3_best_split_ties_and_tiny_case():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    y = np.array([0, 0, 1, 1])
    j, thr, g = lab.best_split(X, y)
    assert (j, thr, g) == (0, 2.5, 0.0)  # both features are perfect; first one wins


def test_task4_depth_sweep_peaks_at_three_on_wine():
    X, y = load_wine(return_X_y=True)
    best, table = lab.best_depth(X, y, depths=(1, 2, 3, 4, 6, 8))
    assert best == 3
    assert list(table.columns) == ["max_depth", "train_acc", "cv_acc", "cv_std"]
    assert table["max_depth"].tolist() == [1, 2, 3, 4, 6, 8]
    assert table["train_acc"].is_monotonic_increasing
    assert table.loc[table["max_depth"] == 3, "cv_acc"].item() > 0.9
    assert table["train_acc"].iloc[-1] == pytest.approx(1.0)
    # reproducible
    best2, table2 = lab.best_depth(X, y, depths=(1, 2, 3, 4, 6, 8))
    assert best2 == best and np.allclose(table["cv_acc"], table2["cv_acc"])


def test_task5_pruning_shrinks_the_tree_and_improves_test_accuracy():
    X, y = make_classification(n_samples=800, n_features=20, n_informative=6, n_redundant=4,
                               flip_y=0.08, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    alpha, pruned, full = lab.prune_alpha(X_tr, y_tr, seed=0)
    assert alpha > 0
    assert full.get_n_leaves() > 40
    assert pruned.get_n_leaves() <= 20
    assert pruned.get_params()["ccp_alpha"] == pytest.approx(alpha)
    assert pruned.score(X_te, y_te) > full.score(X_te, y_te) + 0.03
    assert pruned.score(X_te, y_te) > 0.8


def test_task6_permutation_importance_flags_the_noise_column(loans):
    X, y, X_tr, X_te, y_tr, y_te = loans
    deep = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    rep = lab.importance_report(deep, X_te, y_te, lab.FEATURES, n_repeats=20, seed=0)
    assert list(rep.columns) == ["impurity", "permutation", "suspect"]
    assert set(rep.index) == set(lab.FEATURES)
    assert rep["impurity"].is_monotonic_decreasing
    assert rep["impurity"].sum() == pytest.approx(1.0)
    # the classic mistake: impurity importance makes random_id look real
    assert rep.loc["random_id", "impurity"] > 0.05
    assert bool(rep.loc["random_id", "suspect"]) is True
    assert rep.loc["random_id", "permutation"] < 0.005
    # the real drivers survive
    assert bool(rep.loc["late_payments_24m", "suspect"]) is False
    assert bool(rep.loc["utilization", "suspect"]) is False
    assert rep.loc["late_payments_24m", "permutation"] > 0.02


def test_task7_segment_table_is_five_readable_rules(loans):
    X, y, *_ = loans
    seg = lab.segment_table(X, y, lab.FEATURES, n_segments=5, min_leaf_frac=0.02)
    assert list(seg.columns) == ["rule", "n", "positive_rate"]
    assert len(seg) == 5
    assert seg["n"].sum() == len(y)
    assert (seg["n"] >= 30).all()
    assert seg["positive_rate"].is_monotonic_decreasing
    assert seg["positive_rate"].between(0, 1).all()
    assert seg["positive_rate"].iloc[0] > 0.5 and seg["positive_rate"].iloc[-1] < 0.25
    for rule in seg["rule"]:
        assert ("<=" in rule) or (">" in rule)
        assert any(f in rule for f in lab.FEATURES)
    assert "late_payments_24m" in seg["rule"].iloc[0]
    # the rate in the table is what the data says for that rule
    top = seg.iloc[0]
    mask = np.ones(len(y), dtype=bool)
    for cond in top["rule"].split(" and "):
        name, op, val = cond.split(" ")
        mask &= (X[name] <= float(val)) if op == "<=" else (X[name] > float(val))
    assert abs(mask.sum() - top["n"]) <= 0.01 * len(y)          # thresholds are printed with 3 significant digits
    assert y[mask].mean() == pytest.approx(top["positive_rate"], abs=0.03)
