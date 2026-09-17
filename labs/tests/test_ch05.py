import numpy as np
import pytest

from labs import ch05_linear_regression as lab


def test_task1_mse():
    assert lab.mse([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)
    assert lab.mse(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.0
    assert isinstance(lab.mse([1.0], [0.0]), float)


def test_task2_gradient_matches_finite_differences():
    dw, db = lab.gradient(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), np.array([0.0]), 0.0)
    assert np.allclose(dw, [-5.0]) and db == pytest.approx(-3.0)
    X, y = lab.make_flat_data()
    rng = np.random.default_rng(3)
    w, b = rng.normal(size=3), 1.5
    dw, db = lab.gradient(X, y, w, b)
    assert dw.shape == (3,)
    h = 1e-4
    for j in range(3):
        e = np.zeros(3); e[j] = h
        num = (lab.mse(y, X @ (w + e) + b) - lab.mse(y, X @ (w - e) + b)) / (2 * h)
        assert dw[j] == pytest.approx(num, rel=1e-4)
    num_b = (lab.mse(y, X @ w + b + h) - lab.mse(y, X @ w + b - h)) / (2 * h)
    assert db == pytest.approx(num_b, rel=1e-4)


def test_task3_batch_gd_loss_history_decreases_to_the_optimum():
    X, y = lab.make_flat_data()
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    w, b, losses = lab.batch_gradient_descent(Z, y, lr=0.1, n_epochs=200)
    assert len(losses) == 200
    assert all(b2 <= a for a, b2 in zip(losses, losses[1:]))
    assert losses[0] > 10 * losses[-1]
    A = np.column_stack([np.ones(len(y)), Z])
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    assert losses[-1] == pytest.approx(lab.mse(y, A @ theta), rel=0.01)
    assert np.allclose(w, theta[1:], atol=0.5) and b == pytest.approx(theta[0], abs=0.5)


def test_task4_normal_equation_matches_sklearn():
    from sklearn.linear_model import LinearRegression

    w, b = lab.normal_equation(np.array([[1.0], [2.0], [3.0]]), np.array([3.0, 5.0, 7.0]))
    assert np.allclose(w, [2.0]) and b == pytest.approx(1.0)
    X, y = lab.make_flat_data()
    w, b = lab.normal_equation(X, y)
    ref = LinearRegression().fit(X, y)
    assert np.allclose(w, ref.coef_, atol=1e-6) and b == pytest.approx(ref.intercept_, abs=1e-6)
    # the true generating slopes (2.6, 8, -9) are recovered within noise
    assert abs(w[0] - 2.6) < 0.2 and abs(w[2] + 9.0) < 1.5


def test_task5_standardize_uses_training_statistics_only():
    Z_tr, Z_te, mu, sd = lab.standardize(np.array([[0.0], [2.0]]), np.array([[4.0]]))
    assert Z_tr.tolist() == [[-1.0], [1.0]] and Z_te.tolist() == [[3.0]]
    assert mu.tolist() == [1.0] and sd.tolist() == [1.0]
    X, _ = lab.make_flat_data()
    X_tr, X_te = X[:200], X[200:]
    Z_tr, Z_te, mu, sd = lab.standardize(X_tr, X_te)
    assert np.allclose(Z_tr.mean(axis=0), 0, atol=1e-9) and np.allclose(Z_tr.std(axis=0), 1, atol=1e-9)
    assert np.allclose(mu, X_tr.mean(axis=0)) and np.allclose(sd, X_tr.std(axis=0))
    # the classic mistake: statistics from the test rows (or from all rows) would centre Z_te exactly
    assert not np.allclose(Z_te.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(Z_te, (X_te - X_tr.mean(axis=0)) / X_tr.std(axis=0))


def test_task5_gd_converges_only_after_standardizing():
    X, y = lab.make_flat_data()
    with np.errstate(over="ignore", invalid="ignore"):
        _, _, raw_losses = lab.batch_gradient_descent(X, y, lr=0.1, n_epochs=50)
    assert not np.isfinite(raw_losses[-1]) or raw_losses[-1] > raw_losses[0]      # diverges
    Z, _, _, _ = lab.standardize(X, X)
    _, _, z_losses = lab.batch_gradient_descent(Z, y, lr=0.1, n_epochs=300)
    w, b = lab.normal_equation(Z, y)
    assert z_losses[-1] == pytest.approx(lab.mse(y, Z @ w + b), rel=0.005)


def test_task6_regression_metrics_match_sklearn_and_beat_baseline():
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    m = lab.regression_metrics([1, 2, 3], [1, 2, 5])
    assert set(m) == {"rmse", "mae", "r2"}
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3)) and m["mae"] == pytest.approx(2 / 3) and m["r2"] == pytest.approx(-1.0)
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    Z_tr, Z_te, _, _ = lab.standardize(X_tr, X_te)
    w, b = lab.normal_equation(Z_tr, y_tr)
    pred = Z_te @ w + b
    model = lab.regression_metrics(y_te, pred)
    assert model["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y_te, pred)))
    assert model["mae"] == pytest.approx(mean_absolute_error(y_te, pred))
    assert model["r2"] == pytest.approx(r2_score(y_te, pred))
    base = lab.regression_metrics(y_te, np.full_like(y_te, y_tr.mean()))
    assert abs(base["r2"]) < 0.01                       # the mean baseline explains nothing
    assert 0.30 < model["r2"] < 0.45                    # the chapter's ~0.36
    assert model["rmse"] < base["rmse"] - 10 and model["mae"] < base["mae"] - 10


def test_task7_polynomial_sweep_detects_overfitting():
    x_tr, y_tr, x_te, y_te = lab.make_curve_data()
    degrees = (1, 2, 3, 5, 9, 15)
    results, best = lab.polynomial_sweep(x_tr, y_tr, x_te, y_te, degrees=degrees)
    assert set(results) == set(degrees)
    train = [results[d]["train_mse"] for d in degrees]
    test = [results[d]["test_mse"] for d in degrees]
    assert all(b <= a + 1e-9 for a, b in zip(train, train[1:]))     # training error only falls
    assert best in (3, 5)
    assert best == min(degrees, key=lambda d: results[d]["test_mse"])
    assert results[15]["train_mse"] < results[best]["train_mse"]    # the trap: best on train...
    assert results[15]["test_mse"] > 20 * results[best]["test_mse"]  # ...catastrophic on test
    assert results[1]["test_mse"] > 2 * results[best]["test_mse"]    # and degree 1 underfits
