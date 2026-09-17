import numpy as np
import pytest

from labs import ch01_first_model as lab


def test_task1_baseline_accuracy():
    assert lab.baseline_accuracy(np.array([0, 0, 0, 1])) == pytest.approx(0.75)
    assert lab.baseline_accuracy(np.array([1, 1, 0, 0])) == pytest.approx(0.5)
    _, y = lab.make_fruit_data()
    assert lab.baseline_accuracy(y) == pytest.approx(0.5)


def test_task2_threshold_predict_is_vectorized():
    x = np.array([100.0, 180.0, 181.0, 250.0])
    out = lab.threshold_predict(x, 180.0)
    assert isinstance(out, np.ndarray) and out.dtype.kind == "i"
    assert out.tolist() == [0, 0, 1, 1]


def test_task3_best_threshold_finds_the_optimum():
    x = np.array([120, 130, 140, 200, 210, 220])
    y = np.array([0, 0, 0, 1, 1, 1])
    thr, acc = lab.best_threshold(x, y, candidates=range(100, 260, 10))
    assert acc == pytest.approx(1.0)
    assert 140 <= thr < 200
    # first-on-ties rule
    assert thr == 140


def test_task3_on_realistic_data_beats_baseline():
    x, y = lab.make_fruit_data()
    thr, acc = lab.best_threshold(x, y, candidates=range(100, 300))
    assert acc > lab.baseline_accuracy(y) + 0.3
    assert 160 <= thr <= 200


def test_task4_holdout_split_has_no_leakage():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    X_tr, X_te, y_tr, y_te = lab.holdout_split(X, y, test_frac=0.25, seed=3)
    assert len(y_te) == 5 and len(y_tr) == 15
    assert set(y_tr) | set(y_te) == set(range(20))
    assert set(y_tr) & set(y_te) == set()
    # rows stay aligned with their labels
    for row, label in zip(X_te, y_te):
        assert row.tolist() == [2 * label, 2 * label + 1]
    # seeded -> reproducible
    _, _, _, y_te2 = lab.holdout_split(X, y, test_frac=0.25, seed=3)
    assert y_te.tolist() == y_te2.tolist()


def test_task5_model_beats_baseline_on_unseen_data():
    res = lab.beat_the_baseline(seed=0)
    assert set(res) == {"baseline", "model"}
    assert 0.55 < res["baseline"] < 0.70
    assert res["model"] > 0.93
    assert res["model"] > res["baseline"] + 0.2
