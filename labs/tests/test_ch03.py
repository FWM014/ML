import numpy as np
import pytest

from labs import ch03_math_toolkit as lab


def test_task1_cosine_similarity_ignores_length():
    assert lab.cosine_similarity([1, 2, 0], [10, 20, 0]) == pytest.approx(1.0)
    assert lab.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert lab.cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert lab.cosine_similarity([1.0, 2.0], [4.0, 6.0]) == pytest.approx(0.992, abs=1e-3)
    E = lab.make_product_embeddings()
    sims = np.array([lab.cosine_similarity(E[0], e) for e in E])
    assert sims[0] == pytest.approx(1.0)
    assert np.all(sims >= -1.0 - 1e-9) and np.all(sims <= 1.0 + 1e-9)


def test_task2_batch_predict_values_and_shape():
    rng = np.random.default_rng(0)
    X, W, b = rng.normal(size=(7, 4)), rng.normal(size=(4, 3)), rng.normal(size=3)
    out = lab.batch_predict(X, W, b)
    assert out.shape == (7, 3)
    for i in range(7):
        for j in range(3):
            assert out[i, j] == pytest.approx(float(np.dot(X[i], W[:, j]) + b[j]))
    assert lab.batch_predict([[1, 2]], [[1, 0], [0, 1]], [10, 20]).tolist() == [[11, 22]]


def test_task2_batch_predict_refuses_bad_shapes():
    rng = np.random.default_rng(0)
    X, W, b = rng.normal(size=(5, 3)), rng.normal(size=(3, 4)), rng.normal(size=4)
    with pytest.raises(ValueError) as e:
        lab.batch_predict(X, W.T, b)                    # (5,3) @ (4,3): inner dims 3 vs 4
    assert "(5, 3)" in str(e.value) and "(4, 3)" in str(e.value)
    with pytest.raises(ValueError):
        lab.batch_predict(X, W, b[:2])                   # bias length 2 for 4 models


def test_task3_bayes_posterior_chapter_numbers():
    res = lab.bayes_posterior(0.01, 0.99, 0.95)
    assert res["p_positive"] == pytest.approx(0.0594, abs=1e-4)
    assert res["posterior"] == pytest.approx(0.1667, abs=1e-3)
    assert res["per_1000"]["true_positives"] == pytest.approx(9.9, abs=0.05)
    assert res["per_1000"]["false_positives"] == pytest.approx(49.5, abs=0.05)
    # the quiz: 0.5% fraud, sensitivity 90%, specificity 99% -> about 31% of flags are real
    fraud = lab.bayes_posterior(0.005, 0.90, 0.99)
    assert 0.29 < fraud["posterior"] < 0.33
    # a perfect test on a common condition
    assert lab.bayes_posterior(0.5, 1.0, 1.0)["posterior"] == pytest.approx(1.0)


def test_task4_entropy_in_bits():
    assert lab.entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert lab.entropy([1.0]) == pytest.approx(0.0)
    assert lab.entropy([1.0, 0.0]) == pytest.approx(0.0)           # 0 log 0 = 0, no NaN
    assert lab.entropy([1 / 8] * 8) == pytest.approx(3.0)
    assert lab.entropy([0.9, 0.1]) == pytest.approx(0.469, abs=1e-3)
    assert lab.entropy([0.26, 0.74]) == pytest.approx(0.827, abs=1e-3)


def test_task5_information_gain():
    y = np.array([1, 1, 0, 0])
    assert lab.information_gain(y, np.array([True, True, False, False])) == pytest.approx(1.0)
    assert lab.information_gain(np.array([1, 0, 1, 0]), np.array([True, True, False, False])) == pytest.approx(0.0)
    # the chapter's worked example: a 26/74 node split into 55% (10/90) and 45% (45/55)
    left = np.array([1] * 10 + [0] * 90)
    right = np.array([1] * 45 + [0] * 55)
    y = np.concatenate([np.repeat(left, 11), np.repeat(right, 9)])   # 1100 + 900 rows
    mask = np.array([True] * 1100 + [False] * 900)
    assert y.mean() == pytest.approx(0.2575)             # 0.55*0.10 + 0.45*0.45
    assert lab.information_gain(y, mask) == pytest.approx(0.117, abs=0.005)
    # an empty side must not crash and gives zero gain
    assert lab.information_gain(np.array([1, 0, 1]), np.array([True, True, True])) == pytest.approx(0.0)


def test_task6_numerical_gradient_matches_analytic():
    w = np.array([-1.0, 2.0])
    g = lab.numerical_gradient(lab.pricing_loss, w)
    assert g.shape == (2,)
    assert np.allclose(g, lab.pricing_grad(w), atol=1e-4)
    assert w.tolist() == [-1.0, 2.0]                     # input not modified
    # the chapter exercise: log-loss of a positive example through the sigmoid, x = 2, w = 0.5
    x = 2.0
    loss = lambda w: -np.log(1 / (1 + np.exp(-w[0] * x)))
    assert lab.numerical_gradient(loss, np.array([0.5]))[0] == pytest.approx(-0.538, abs=1e-3)


def test_task7_gradient_descent_converges_and_losses_fall():
    w, losses = lab.gradient_descent(lab.pricing_loss, lab.pricing_grad, np.array([-1.0, 2.0]), lr=0.1, n_steps=25)
    assert len(losses) == 25
    assert np.allclose(w, [2.0, -1.0], atol=0.02)
    assert all(b < a for a, b in zip(losses, losses[1:]))
    assert losses[0] == pytest.approx(10.08, abs=0.01)   # the chapter's first step
    assert losses[-1] < 1e-3


def test_task7_gradient_check_catches_a_wrong_gradient():
    wrong_grad = lambda w: np.array([2 * (w[0] - 2), 2 * (w[1] + 1)])   # forgot the factor 3
    with pytest.raises(ValueError):
        lab.gradient_descent(lab.pricing_loss, wrong_grad, np.array([-1.0, 2.0]), lr=0.1, n_steps=5)
