import numpy as np
import pytest

from labs import ch06_classification_metrics as lab


def test_task1_sigmoid_values_and_stability():
    out = lab.sigmoid(np.array([-4.0, 0.0, 4.0]))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [0.018, 0.5, 0.982], atol=1e-3)
    with np.errstate(over="raise"):                       # any overflow becomes an error
        big = lab.sigmoid(np.array([-1000.0, 1000.0]))
    assert big[0] == pytest.approx(0.0, abs=1e-12) and big[1] == pytest.approx(1.0)


def test_task2_log_loss_matches_sklearn_and_punishes_confidence():
    from sklearn.metrics import log_loss as sk_log_loss

    assert lab.log_loss([1, 0], [0.9, 0.1]) == pytest.approx(-np.log(0.9))
    assert lab.log_loss([0], [0.999]) == pytest.approx(6.91, abs=0.01)
    assert np.isfinite(lab.log_loss([0, 1], [1.0, 0.0]))    # clipped, not inf
    y, p = lab.make_fraud_scores()
    assert lab.log_loss(y, p) == pytest.approx(sk_log_loss(y, p), rel=1e-6)
    assert lab.log_loss(y, p) < lab.log_loss(y, np.full_like(p, y.mean()))


def test_task3_confusion_counts_match_sklearn():
    from sklearn.metrics import confusion_matrix

    assert lab.confusion_counts([1, 1, 0, 0], [1, 0, 1, 0]) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    y, p = lab.make_fraud_scores()
    pred = (p >= 0.5).astype(int)
    c = lab.confusion_counts(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    assert (c["tn"], c["fp"], c["fn"], c["tp"]) == (tn, fp, fn, tp)
    assert all(isinstance(v, int) for v in c.values())


def test_task4_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y = np.array([1] * 50 + [0] * 950)
    pred = np.array([1] * 35 + [0] * 15 + [1] * 20 + [0] * 930)   # the chapter's table
    m = lab.classification_metrics(y, pred)
    assert m["accuracy"] == pytest.approx(0.965)
    assert m["precision"] == pytest.approx(35 / 55)
    assert m["recall"] == pytest.approx(0.70)
    assert m["specificity"] == pytest.approx(930 / 950)
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    yf, p = lab.make_fraud_scores()
    for t in (0.2, 0.5, 0.8):
        pf = (p >= t).astype(int)
        mf = lab.classification_metrics(yf, pf)
        assert mf["accuracy"] == pytest.approx(accuracy_score(yf, pf))
        assert mf["precision"] == pytest.approx(precision_score(yf, pf))
        assert mf["recall"] == pytest.approx(recall_score(yf, pf))
        assert mf["f1"] == pytest.approx(f1_score(yf, pf))


def test_task4_accuracy_paradox_on_95_5_data():
    """Blocking nothing scores 95% accuracy and catches zero fraud: never lead with accuracy."""
    y, p = lab.make_fraud_scores()
    nothing = lab.classification_metrics(y, np.zeros_like(y))
    assert nothing["accuracy"] > 0.94
    assert nothing["recall"] == 0.0 and nothing["precision"] == 0.0 and nothing["f1"] == 0.0
    model = lab.classification_metrics(y, (p >= 0.5).astype(int))
    assert model["accuracy"] < nothing["accuracy"] + 0.04      # accuracy barely moves...
    assert model["recall"] > 0.5 and model["f1"] > 0.5         # ...while the real metrics jump


def test_task5_roc_auc_rank_statistic_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    # chapter exercise 2: D, D, R, D, R, R, R, R sorted by score -> 14/15
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    s = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    assert lab.roc_auc(y, s) == pytest.approx(14 / 15)
    assert lab.roc_auc([0, 1], [0.9, 0.1]) == pytest.approx(0.0)         # flipped labels
    assert lab.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)   # all tied
    y, p = lab.make_fraud_scores()
    assert lab.roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-9)
    rng = np.random.default_rng(0)
    yt, st = rng.integers(0, 2, 300), rng.integers(0, 5, 300).astype(float)   # many ties
    assert lab.roc_auc(yt, st) == pytest.approx(roc_auc_score(yt, st), abs=1e-9)
    # a rank statistic ignores monotone rescaling
    assert lab.roc_auc(y, np.log(p / (1 - p))) == pytest.approx(lab.roc_auc(y, p))


def test_task6_threshold_by_value_beats_default_and_accuracy_optimum():
    y, p = lab.make_fraud_scores()
    values = {"tp": 0.0, "tn": 0.0, "fp": -5.0, "fn": -200.0}
    t, v = lab.best_threshold_by_value(y, p, values)
    assert 0.01 <= t < 0.35                                  # far below 0.5 when misses are dear
    grid = np.arange(0.01, 1.0, 0.01)

    def value_at(thr):
        c = lab.confusion_counts(y, (p >= thr).astype(int))
        return sum(c[k] * values[k] for k in c)

    def accuracy_at(thr):
        return lab.classification_metrics(y, (p >= thr).astype(int))["accuracy"]

    assert v == pytest.approx(value_at(t))
    assert v >= max(value_at(g) for g in grid) - 1e-9        # truly the best on the grid
    assert v > value_at(0.5) + 1000                          # thousands of euros better than 0.5
    t_acc = grid[int(np.argmax([accuracy_at(g) for g in grid]))]
    assert t_acc > t + 0.1                                    # the accuracy optimum is a different, worse cut
    assert v > value_at(t_acc) + 500
    # ties: first threshold wins; custom grid respected
    t2, v2 = lab.best_threshold_by_value([1, 0], [0.3, 0.2], values, grid=[0.1, 0.25, 0.4])
    assert t2 == 0.25 and v2 == 0.0


def test_task7_tumour_screen_recall_first_threshold():
    from sklearn.pipeline import Pipeline

    res = lab.tumour_screen(seed=42, min_recall=0.98)
    assert isinstance(res["model"], Pipeline)
    assert res["auc"] > 0.98
    assert 0.01 <= res["threshold"] < 0.5
    assert res["metrics"]["recall"] >= 0.97                  # tuned out-of-fold, still holds on test
    assert res["metrics"]["recall"] >= res["metrics_at_0_5"]["recall"]
    assert res["metrics"]["precision"] > 0.75
    assert 0.55 < res["majority_baseline"] < 0.70
    assert res["metrics_at_0_5"]["accuracy"] > res["majority_baseline"] + 0.25
    assert res["threshold"] >= 0.03                          # not the fallback: a real, tuned cut
    # an unattainable recall target falls back to the lowest threshold in the grid
    strict = lab.tumour_screen(seed=42, min_recall=1.0)
    assert strict["threshold"] == pytest.approx(0.01)
    assert strict["metrics"]["recall"] >= res["metrics"]["recall"]
