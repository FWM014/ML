"""
Lab 05 — Linear regression from scratch (Chapter 5: Linear Regression & Gradient Descent)
=========================================================================================
Chapter: 05 — Linear Regression & Gradient Descent

THE PROBLEM
-----------
Maison Verte, a Lyon estate agency, prices flats by gut feeling and loses deals both
ways: overpriced flats sit for months, underpriced ones leave money on the table. You
have 300 past sales with three numbers per flat (size in m², number of rooms, distance
to the city centre in km) and the sale price in k€. The owner will only trust a model
she can read, so the first model is a line: price = w·x + b. You will fit it two ways
(the normal equation and your own gradient descent), prove they agree, show why the
gradient descent only converges once the features are standardised, and report RMSE,
MAE and R² next to the "just predict the average" baseline. Then a clinic partner
asks the same skeleton on a medical dataset (diabetes progression), and a junior
colleague asks whether a degree-15 polynomial "with training R² 0.99" is a good idea.

TASKS (make the tests in labs/tests/test_ch05.py pass, one at a time)
---------------------------------------------------------------------
1. mse(y, y_hat)                              → mean squared error
2. gradient(X, y, w, b)                       → (dL/dw, dL/db) of MSE for ŷ = Xw + b
3. batch_gradient_descent(X, y, lr, epochs)   → (w, b, losses); losses must decrease
4. normal_equation(X, y)                      → closed-form (w, b) via np.linalg.solve
5. standardize(X_train, X_test)               → z-scores using TRAINING statistics only
6. regression_metrics(y, y_hat)               → {"rmse", "mae", "r2"} from scratch
7. polynomial_sweep(x_tr, y_tr, x_te, y_te)   → train/test MSE per degree, best degree by TEST

STRETCH (no tests)
------------------
- Generate prices whose noise grows with size (sd = 0.2*size), fit on y and on log(y), and
  compare the residuals-vs-fitted plots (chapter exercise 2).
- Implement mini-batch gradient descent (batch_size=32) and compare its loss curve with the
  batch version at the same learning rate.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np


def make_flat_data(seed: int = 0, n: int = 300):
    """300 flat sales: X = [size_m2, rooms, km_to_centre], y = price in k€. Do not modify."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(30, 160, n)
    rooms = np.clip(np.round(size / 28 + rng.normal(0, 0.7, n)), 1, 7)
    km = rng.uniform(0.5, 12, n)
    price = 40 + 2.6 * size + 8.0 * rooms - 9.0 * km + rng.normal(0, 25, n)
    X = np.column_stack([size, rooms, km]).round(2)
    return X, price.round(1)


def make_curve_data(seed: int = 1, n_train: int = 20, n_test: int = 200):
    """A non-linear truth (sin) with noise sd 0.3: 20 train points, 200 test. Do not modify."""
    rng = np.random.default_rng(seed)
    f = lambda x: np.sin(2.2 * x)
    x_tr = rng.uniform(-2, 2, n_train); y_tr = f(x_tr) + rng.normal(0, 0.3, n_train)
    x_te = rng.uniform(-2, 2, n_test); y_te = f(x_te) + rng.normal(0, 0.3, n_test)
    return x_tr, y_tr, x_te, y_te


# ---------------------------------------------------------------- Task 1
def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error (chapter §2): mean of (y - ŷ)².

    Example: y = [1, 2, 3], y_hat = [1, 2, 5] -> 4/3 ≈ 1.333
    """
    # TODO: np.mean((y - y_hat) ** 2), returned as a float
    raise NotImplementedError("Task: mse")


# ---------------------------------------------------------------- Task 2
def gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """Partial derivatives of MSE for ŷ = X @ w + b (chapter §4 "the derivatives, step by step").

    With err = ŷ - y (shape (n,)):
        dL/dw = (2/n) * X.T @ err      shape (d,)
        dL/db = (2/n) * sum(err)       float
    X is (n, d), w is (d,). Return (dw, db).

    Example: X = [[1], [2]], y = [1, 2], w = [0], b = 0 -> err = [-1, -2],
             dw = [2/2 * (1*-1 + 2*-2)] = [-5.0], db = 2/2 * (-3) = -3.0
    """
    # TODO: err = X @ w + b - y; dw = 2/n * X.T @ err; db = 2/n * err.sum()
    raise NotImplementedError("Task: gradient")


# ---------------------------------------------------------------- Task 3
def batch_gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_epochs: int = 200
                           ) -> tuple[np.ndarray, float, list[float]]:
    """Batch gradient descent from w = 0, b = 0 (chapter §4 update rule, §5 "batch").

    Each epoch: compute the gradient on ALL rows with gradient(), then
        w <- w - lr * dw ;  b <- b - lr * db
    and append mse(y, X @ w + b) AFTER the update to `losses`.
    Returns (w, b, losses) with len(losses) == n_epochs. On standardised features with a
    sane learning rate the losses decrease monotonically toward the normal-equation loss.

    Example: X = standardised size column of the flat data, lr=0.1, 100 epochs ->
             losses[0] > losses[1] > ... and losses[-1] within 1% of the closed-form MSE.
    """
    # TODO: start at np.zeros(X.shape[1]) and 0.0; loop n_epochs: call gradient(), update w and b, append mse() after the update
    raise NotImplementedError("Task: batch_gradient_descent")


# ---------------------------------------------------------------- Task 4
def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """The closed form (chapter §3): θ* = (AᵀA)⁻¹Aᵀy with A = [1 | X].

    Build the design matrix A = np.column_stack([np.ones(n), X]) and solve
    np.linalg.solve(A.T @ A, A.T @ y). The first entry of θ is b, the rest is w.
    Returns (w, b) matching sklearn's LinearRegression coef_ and intercept_.

    Example: X = [[1], [2], [3]], y = [3, 5, 7] -> w = [2.0], b = 1.0
    """
    # TODO: A = np.column_stack([np.ones(n), X]); theta = np.linalg.solve(A.T @ A, A.T @ y); split theta into b (first) and w (rest)
    raise NotImplementedError("Task: normal_equation")


# ---------------------------------------------------------------- Task 5
def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """z = (x - mu) / sd per column, with mu and sd computed on X_train ONLY (chapter §5, §9).

    Returns (Z_train, Z_test, mu, sd). Z_test uses the training mu/sd: fitting the scaler on
    test rows (or on everything) is the leak the chapter warns about. Use X_train.std(axis=0)
    (population std, ddof=0), as StandardScaler does.

    Example: X_train = [[0], [2]], X_test = [[4]] -> Z_train = [[-1], [1]], Z_test = [[3]], mu=[1], sd=[1]
    """
    # TODO: mu = X_train.mean(axis=0); sd = X_train.std(axis=0); apply (X - mu) / sd to BOTH arrays with the same mu, sd
    raise NotImplementedError("Task: standardize")


# ---------------------------------------------------------------- Task 6
def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict:
    """RMSE, MAE and R² from scratch (chapter §8), matching sklearn.metrics.

    rmse = sqrt(mean((y - ŷ)²)); mae = mean(|y - ŷ|);
    r2 = 1 - sum((y - ŷ)²) / sum((y - mean(y))²)   (0 for the mean baseline, negative if worse).
    Returns {"rmse": float, "mae": float, "r2": float}.

    Example: y = [1, 2, 3], y_hat = [1, 2, 5] -> rmse 1.155, mae 0.667, r2 = 1 - 4/2 = -1.0
    """
    # TODO: resid = y - y_hat; rmse = sqrt(mean(resid**2)); mae = mean(|resid|); r2 = 1 - sum(resid**2) / sum((y - y.mean())**2)
    raise NotImplementedError("Task: regression_metrics")


# ---------------------------------------------------------------- Task 7
def polynomial_sweep(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
                     degrees=(1, 2, 3, 5, 9, 15)) -> tuple[dict, int]:
    """Fit make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()) per degree (§7).

    x_tr / x_te are 1-D; reshape to (n, 1) before fitting. For each degree record
    {"train_mse": ..., "test_mse": ...} (use sklearn.metrics.mean_squared_error or mse()).
    Returns (results, best_degree) where results = {degree: {...}} and best_degree is the
    degree with the LOWEST TEST MSE. The training MSE keeps falling with degree; the test
    MSE does not. That gap is overfitting, and the junior's "training R² 0.99" is the trap.

    Example: on make_curve_data() -> best_degree in (3, 5); degree 15 test_mse >> 1.
    """
    # TODO: reshape x[:, None]; for each degree fit make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()); record both MSEs; best = min over test_mse
    raise NotImplementedError("Task: polynomial_sweep")


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = make_flat_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    w_ne, b_ne = normal_equation(X_tr, y_tr)
    print("closed form on raw features: w =", w_ne.round(2), "b =", round(b_ne, 1))
    with np.errstate(over="ignore", invalid="ignore"):          # we expect this one to blow up
        _, _, losses_raw = batch_gradient_descent(X_tr, y_tr, lr=0.1, n_epochs=50)
    print(f"GD on RAW features, lr=0.1: loss after 50 epochs = {losses_raw[-1]:.3g}  (diverged)")
    Z_tr, Z_te, mu, sd = standardize(X_tr, X_te)
    w_gd, b_gd, losses = batch_gradient_descent(Z_tr, y_tr, lr=0.1, n_epochs=300)
    w_z, b_z = normal_equation(Z_tr, y_tr)
    print(f"GD on standardised features: loss {losses[0]:.1f} -> {losses[-1]:.2f}; closed form {mse(y_tr, Z_tr @ w_z + b_z):.2f}")
    print("slope per m² recovered from GD:", round(w_gd[0] / sd[0], 3), "(closed form on raw:", round(w_ne[0], 3), ")")
    print("test metrics :", {k: round(v, 3) for k, v in regression_metrics(y_te, Z_te @ w_gd + b_gd).items()})
    print("mean baseline:", {k: round(v, 3) for k, v in regression_metrics(y_te, np.full_like(y_te, y_tr.mean())).items()})

    Xd, yd = load_diabetes(return_X_y=True)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.25, random_state=0)
    Zd_tr, Zd_te, _, _ = standardize(Xd_tr, Xd_te)
    wd, bd = normal_equation(Zd_tr, yd_tr)
    print("\ndiabetes model   :", {k: round(v, 3) for k, v in regression_metrics(yd_te, Zd_te @ wd + bd).items()})
    print("diabetes baseline:", {k: round(v, 3) for k, v in regression_metrics(yd_te, np.full_like(yd_te, yd_tr.mean())).items()})

    res, best = polynomial_sweep(*make_curve_data())
    print("\ndegree   train MSE   test MSE")
    for d, r in res.items():
        print(f"{d:6d}   {r['train_mse']:9.3f}   {r['test_mse']:9.3f}")
    print("best degree by TEST error:", best)
