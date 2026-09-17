import numpy as np
import pytest

from labs import ch15_neural_net_numpy as lab


# ---------------------------------------------------------------- Task 1
def test_task1_activations_and_derivatives():
    z = np.array([-2.0, 0.0, 2.0])
    assert lab.sigmoid(z) == pytest.approx([0.1192, 0.5, 0.8808], abs=1e-4)
    assert lab.relu(np.array([-2.0, 0.0, 3.0])).tolist() == [0.0, 0.0, 3.0]
    assert lab.sigmoid_grad(z) == pytest.approx([0.105, 0.25, 0.105], abs=1e-3)
    assert lab.relu_grad(np.array([-1.0, 0.0, 2.0])).tolist() == [0.0, 0.0, 1.0]
    assert lab.sigmoid_grad(np.linspace(-10, 10, 201)).max() == pytest.approx(0.25)
    # derivative must match a finite difference
    zz = np.linspace(-4, 4, 17)
    eps = 1e-6
    num = (lab.sigmoid(zz + eps) - lab.sigmoid(zz - eps)) / (2 * eps)
    assert np.allclose(lab.sigmoid_grad(zz), num, atol=1e-6)


def test_task1_sigmoid_is_numerically_stable():
    with np.errstate(over="raise"):
        big = lab.sigmoid(np.array([-800.0, 800.0]))
    assert big.tolist() == [0.0, 1.0]
    assert lab.sigmoid(np.array([[0.0]])).shape == (1, 1)


# ---------------------------------------------------------------- Task 2
def test_task2_init_params_shapes_scales_and_seed():
    p = lab.init_params(4, 3, 2, seed=0)
    assert set(p) == {"W1", "b1", "W2", "b2"}
    assert p["W1"].shape == (4, 3) and p["b1"].shape == (3,)
    assert p["W2"].shape == (3, 2) and p["b2"].shape == (2,)
    assert np.all(p["b1"] == 0) and np.all(p["b2"] == 0)
    big = lab.init_params(200, 100, 1, seed=1)
    assert big["W1"].std() == pytest.approx(np.sqrt(2 / 200), rel=0.1)
    assert big["W2"].std() == pytest.approx(np.sqrt(1 / 100), rel=0.15)
    assert np.array_equal(lab.init_params(4, 3, 2, seed=7)["W1"], lab.init_params(4, 3, 2, seed=7)["W1"])
    assert not np.array_equal(lab.init_params(4, 3, 2, seed=7)["W1"], lab.init_params(4, 3, 2, seed=8)["W1"])


# ---------------------------------------------------------------- Task 3
def test_task3_count_parameters():
    assert lab.count_parameters(lab.init_params(4, 3, 2)) == 23
    assert lab.count_parameters(lab.init_params(784, 128, 10)) == 101770
    assert lab.count_parameters(lab.init_params(2, 16, 1)) == 2 * 16 + 16 + 16 + 1


# ---------------------------------------------------------------- Task 4
def test_task4_forward_shapes_values_and_input_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4))
    p = lab.init_params(4, 3, 1, seed=0)
    A2, cache = lab.forward(X, p)
    assert A2.shape == (5, 1)
    assert cache["Z1"].shape == cache["A1"].shape == (5, 3)
    assert cache["Z2"].shape == cache["A2"].shape == (5, 1)
    assert np.all((A2 > 0) & (A2 < 1))
    Z1 = X @ p["W1"] + p["b1"]
    assert np.allclose(cache["Z1"], Z1)
    assert np.allclose(cache["A1"], np.maximum(0, Z1))
    assert np.allclose(A2, 1 / (1 + np.exp(-(np.maximum(0, Z1) @ p["W2"] + p["b2"]))))
    # sigmoid hidden option
    A2s, cs = lab.forward(X, p, hidden="sigmoid")
    assert np.allclose(cs["A1"], 1 / (1 + np.exp(-Z1)))
    # classic bug: transposed / wrong-width input must not silently broadcast
    with pytest.raises(ValueError):
        lab.forward(X.T, p)


# ---------------------------------------------------------------- Task 5
def test_task5_bce_loss_values_and_safety():
    assert lab.bce_loss(np.full((4, 1), 0.5), np.array([[0], [1], [0], [1]])) == pytest.approx(np.log(2), abs=1e-6)
    assert lab.bce_loss(np.array([[0.9], [0.1]]), np.array([[1], [0]])) == pytest.approx(-np.log(0.9), abs=1e-6)
    assert lab.bce_loss(np.array([0.2, 0.8]), np.array([1, 0])) == pytest.approx(-np.log(0.2), abs=1e-6)
    assert isinstance(lab.bce_loss(np.array([[0.5]]), np.array([[1]])), float)
    # exactly 0 or 1 predictions must stay finite
    assert np.isfinite(lab.bce_loss(np.array([[1.0], [0.0]]), np.array([[0], [1]])))
    # a (n,) y against an (n, 1) y_hat must not broadcast into an (n, n) mean
    assert lab.bce_loss(np.array([[0.9], [0.1]]), np.array([1, 0])) == pytest.approx(-np.log(0.9), abs=1e-6)


# ---------------------------------------------------------------- Task 6
def test_task6_backward_matches_the_worked_example():
    p = {"W1": np.array([[0.5]]), "b1": np.zeros(1), "W2": np.array([[-1.0]]), "b2": np.zeros(1)}
    X, Y = np.array([[1.0]]), np.array([[1.0]])
    _, cache = lab.forward(X, p, hidden="sigmoid")
    g = lab.backward(X, Y, p, cache, hidden="sigmoid")
    assert g["dW2"] == pytest.approx(np.array([[-0.4051]]), abs=1e-4)
    assert g["db2"] == pytest.approx(np.array([-0.6508]), abs=1e-4)
    assert g["dW1"] == pytest.approx(np.array([[0.1529]]), abs=1e-4)
    assert g["db1"] == pytest.approx(np.array([0.1529]), abs=1e-4)


def test_task6_backward_passes_a_numerical_gradient_check():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 3))
    Y = rng.integers(0, 2, size=(12, 1)).astype(float)
    for hidden in ["relu", "sigmoid"]:
        p = lab.init_params(3, 5, 1, seed=4)
        _, cache = lab.forward(X, p, hidden=hidden)
        g = lab.backward(X, Y, p, cache, hidden=hidden)
        for k in ["W1", "b1", "W2", "b2"]:
            assert g["d" + k].shape == p[k].shape
            num = lab.numerical_gradient(lambda q: lab.bce_loss(lab.forward(X, q, hidden=hidden)[0], Y), p, k)
            assert np.abs(g["d" + k] - num).max() < 1e-5, (hidden, k)


# ---------------------------------------------------------------- Task 7
def test_task7_training_reaches_90_percent_on_unseen_moons():
    X_tr, X_te, y_tr, y_te = lab.make_churn_data(seed=0)
    params, hist = lab.train_mlp(X_tr, y_tr, n_hidden=16, lr=1.0, epochs=1000, seed=1)
    assert len(hist) == 1000
    assert hist[0] > 0.5 and hist[-1] < 0.15
    assert hist[-1] < hist[0]
    pred = lab.predict(X_te, params)
    assert pred.shape == (len(y_te),) and set(np.unique(pred)) <= {0, 1}
    assert (pred == y_te).mean() > 0.9
    assert (lab.predict(X_tr, params) == y_tr).mean() > 0.92
    # seeded -> reproducible
    _, hist2 = lab.train_mlp(X_tr, y_tr, n_hidden=16, lr=1.0, epochs=1000, seed=1)
    assert hist2[-1] == pytest.approx(hist[-1])


def test_task7_more_hidden_units_still_generalise():
    X_tr, X_te, y_tr, y_te = lab.make_churn_data(seed=2)
    params, hist = lab.train_mlp(X_tr, y_tr, n_hidden=32, lr=0.5, epochs=800, seed=3)
    assert lab.count_parameters(params) == 2 * 32 + 32 + 32 + 1
    assert (lab.predict(X_te, params) == y_te).mean() > 0.88
