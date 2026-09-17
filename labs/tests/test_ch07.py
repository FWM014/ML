import numpy as np
import pytest

from labs import ch07_generalization as lab


@pytest.fixture(scope="module")
def cancer():
    from sklearn.datasets import load_breast_cancer
    return load_breast_cancer(return_X_y=True)


@pytest.fixture(scope="module")
def diabetes():
    from sklearn.datasets import load_diabetes
    return load_diabetes(return_X_y=True)


def test_task1_diagnose_rule_matches_chapter_exercise():
    assert lab.diagnose(0.99, 0.88) == "high variance"       # gap 11 points
    assert lab.diagnose(0.86, 0.85) == "high bias"           # both low, tiny gap
    assert lab.diagnose(0.995, 0.97) == "ok"                 # under target, small gap
    assert lab.diagnose(0.85, 0.70) == "high variance"       # both: fix variance first
    assert lab.diagnose(0.90, 0.88, target=0.85) == "ok"     # thresholds are parameters
    assert lab.diagnose(0.90, 0.88, target=0.85, gap_tol=0.01) == "high variance"


def test_task2_degree_sweep_picks_by_validation_not_training():
    X, y = lab.make_wave_data()
    degrees = (1, 2, 3, 5, 9, 12)
    res, best = lab.degree_sweep(X, y, degrees=degrees, seed=0)
    assert set(res) == set(degrees)
    train = [res[d]["train_mse"] for d in degrees]
    val = [res[d]["val_mse"] for d in degrees]
    assert all(b <= a + 1e-9 for a, b in zip(train, train[1:]))      # training error only falls
    assert best in (3, 5)
    assert best == min(degrees, key=lambda d: res[d]["val_mse"])
    assert res[12]["train_mse"] < res[best]["train_mse"]              # the memo's mistake...
    assert res[12]["val_mse"] > res[best]["val_mse"] * 1.3            # ...is punished by validation
    assert res[1]["val_mse"] > 2 * res[best]["val_mse"]               # degree 1 underfits
    assert 0.08 < res[best]["val_mse"] < 0.16                         # near the 0.09 noise floor
    res2, _ = lab.degree_sweep(X, y, degrees=degrees, seed=0)
    assert res2 == res                                                # seeded -> reproducible


def test_task3_ridge_shrinkage(diabetes):
    X, y = diabetes
    alphas = (0.01, 1, 10, 100)
    out = lab.ridge_shrinkage(X, y, alphas=alphas)
    norms = [out[a]["l2_norm"] for a in alphas]
    assert all(b < a for a, b in zip(norms, norms[1:]))               # strictly shrinking
    assert 60 < norms[0] < 70 and 30 < norms[-1] < 40                 # the chapter's 65.4 -> 34.8
    for a in alphas:
        assert out[a]["coef"].shape == (10,)
        assert out[a]["l2_norm"] == pytest.approx(np.linalg.norm(out[a]["coef"]))
    # the coefficients are on the standardised scale (bmi ≈ 24.7 at tiny alpha), not raw (≈ 520)
    assert 20 < out[0.01]["coef"][2] < 30


def test_task4_lasso_sparsity(diabetes):
    X, y = diabetes
    alphas = (0.01, 1, 5, 20)
    out = lab.lasso_sparsity(X, y, alphas=alphas)
    counts = [out[a] for a in alphas]
    assert all(isinstance(c, int) for c in counts)
    assert counts[0] == 10                                            # nothing switched off yet
    assert all(b <= a for a, b in zip(counts, counts[1:]))
    assert counts[-1] <= 4 and counts[-1] >= 1                        # most features off, not all
    assert counts[1] < counts[0]


def test_task5_cv_uses_a_pipeline_so_the_scaler_cannot_leak(cancer):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = cancer
    res = lab.cv_accuracy(X, y, C=0.1, seed=0)
    model = res["model"]
    assert isinstance(model, Pipeline), "the scaler must live inside a Pipeline"
    steps = [s for _, s in model.steps]
    assert any(isinstance(s, StandardScaler) for s in steps)
    assert isinstance(steps[-1], LogisticRegression) and steps[-1].C == 0.1
    assert len(res["scores"]) == 5
    assert res["mean"] == pytest.approx(np.mean(res["scores"])) and res["std"] == pytest.approx(np.std(res["scores"]))
    assert 0.96 < res["mean"] < 0.99 and res["std"] < 0.03
    # exactly what cross_val_score gives with the stated splitter (reproducible protocol)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ref = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    assert np.allclose(np.sort(res["scores"]), np.sort(ref))
    # the pipeline must not have been fitted on all rows behind the scenes
    assert not hasattr(steps[0], "mean_")


def test_task6_learning_curve_summary_separates_variance_from_bias(cancer):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y = cancer
    tree = lab.learning_curve_summary(DecisionTreeClassifier(random_state=0), X, y, seed=0)
    assert tree["n_train"] == [45, 113, 227, 341, 455]
    assert len(tree["train"]) == len(tree["val"]) == 5
    assert all(t == pytest.approx(1.0) for t in tree["train"])       # memorises every size
    assert tree["gap_at_max"] == pytest.approx(tree["train"][-1] - tree["val"][-1])
    assert 0.04 < tree["gap_at_max"] < 0.12                          # textbook variance
    logit = lab.learning_curve_summary(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)), X, y, seed=0)
    assert logit["gap_at_max"] < 0.02                                # curves meet: no variance problem
    assert logit["val"][-1] > 0.96
    assert logit["val_still_rising"] is False                        # plateaued: more data will not help
    assert isinstance(tree["val_still_rising"], bool)


def test_task7_model_report_recommends_the_generalising_model(cancer):
    X, y = cancer
    rep = lab.model_report(X, y, seed=0)
    assert set(rep["candidates"]) == {"logistic", "deep_tree"}
    tree, logit = rep["candidates"]["deep_tree"], rep["candidates"]["logistic"]
    assert tree["train_score"] == pytest.approx(1.0)
    assert tree["cv_mean"] < 0.95
    assert tree["diagnosis"] == "high variance"
    assert logit["cv_mean"] > 0.96 and logit["train_score"] - logit["cv_mean"] < 0.03
    assert logit["diagnosis"] == "ok"
    assert rep["recommended"] == "logistic"
    for c in rep["candidates"].values():
        assert c["diagnosis"] == lab.diagnose(c["train_score"], c["cv_mean"])
