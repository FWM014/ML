"""
Lab 15 — A neural network you can audit (Chapter 15: Neural Networks From Scratch)
==================================================================================

THE PROBLEM
-----------
Brightline Telecom's churn model is a logistic regression. The data-science lead wants
to move to a neural network, but the model-risk committee (this is a regulated
business) will not sign off on "we called model.fit()". They want a network whose every
number can be reproduced by hand: the activations and their derivatives, the forward
pass with explicit shapes, the loss, and — above all — gradients that match a numerical
finite-difference check to 1e-5, because a wrong backward pass trains silently to a
worse model. You will build a 2-layer MLP (ReLU hidden layer, sigmoid output, binary
cross-entropy) in NumPy only, prove the gradients, count parameters for the committee's
memo, and train it on a two-moons stand-in for the churn table to > 90 % held-out
accuracy. Then the PyTorch chapter can automate what you now understand.

TASKS (make the tests in labs/tests/test_ch15.py pass, one at a time)
---------------------------------------------------------------------
1. sigmoid / relu / sigmoid_grad / relu_grad     → activations and their derivatives (§3)
2. init_params(n_in, n_hidden, n_out, seed)      → He-initialised weights, zero biases (§8)
3. count_parameters(params)                      → sum of (in x out + out) per layer (§4)
4. forward(X, params, hidden)                    → (probabilities, cache) with shape checks (§4)
5. bce_loss(y_hat, y)                            → binary cross-entropy, safe at 0 and 1 (§5)
6. backward(X, Y, params, cache, hidden)         → gradients that pass a numerical check (§6)
7. train_mlp(...) / predict(X, params)           → full-batch gradient descent on two moons (§7)

STRETCH (no tests)
------------------
- Replace the ReLU hidden layer with sigmoid in train_mlp and compare the loss curves.
  Then stack 10 hidden layers (§8) and print the gradient norm per layer.
- Add a softmax output with K classes and derive dZ2 = (P - Y_onehot) / n yourself.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def make_churn_data(seed: int = 0, n: int = 600):
    """Two-moons stand-in for the churn table. Returns X_tr, X_te, y_tr, y_te. Do not modify."""
    X, y = make_moons(n_samples=n, noise=0.25, random_state=seed)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def numerical_gradient(loss_of_params, params: dict, key: str, eps: float = 1e-5) -> np.ndarray:
    """Central finite differences of loss_of_params(params) w.r.t. params[key]. Do not modify.

    Slow (one loss evaluation per parameter entry, times two) but trustworthy: this is the
    referee for your backward pass.
    """
    grad = np.zeros_like(params[key], dtype=float)
    it = np.nditer(params[key], flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = params[key][idx]
        params[key][idx] = old + eps
        plus = loss_of_params(params)
        params[key][idx] = old - eps
        minus = loss_of_params(params)
        params[key][idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


# ---------------------------------------------------------------- Task 1
def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-z)), elementwise. sigmoid(0) == 0.5; sigmoid(±40) is 1.0 / 0.0 without
    overflow warnings (hint: np.exp(-np.abs(z)) and a np.where on the sign, or np.clip)."""
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def relu(z: np.ndarray) -> np.ndarray:
    """max(0, z), elementwise. relu([-2, 0, 3]) == [0, 0, 3]."""
    return np.maximum(0, np.asarray(z, dtype=float))


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    """d sigmoid / dz = sigmoid(z) * (1 - sigmoid(z)). Peaks at 0.25 when z == 0 — the
    reason deep sigmoid nets suffer vanishing gradients (§8)."""
    s = sigmoid(z)
    return s * (1 - s)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """d relu / dz = 1 where z > 0 else 0, as a float array. relu_grad([-1, 0, 2]) == [0, 0, 1]."""
    return (np.asarray(z, dtype=float) > 0).astype(float)


# ---------------------------------------------------------------- Task 2
def init_params(n_in: int, n_hidden: int, n_out: int = 1, seed: int = 1) -> dict:
    """He initialisation with rng = np.random.default_rng(seed):

    W1 = rng.normal(size=(n_in, n_hidden)) * sqrt(2 / n_in),   b1 = zeros(n_hidden)
    W2 = rng.normal(size=(n_hidden, n_out)) * sqrt(1 / n_hidden), b2 = zeros(n_out)
    (draw W1 first, then W2). Returns {"W1", "b1", "W2", "b2"} as float arrays.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(size=(n_in, n_hidden)) * np.sqrt(2 / n_in),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(size=(n_hidden, n_out)) * np.sqrt(1 / n_hidden),
        "b2": np.zeros(n_out),
    }


# ---------------------------------------------------------------- Task 3
def count_parameters(params: dict) -> int:
    """Total number of trainable scalars: the sum of .size over every array in params.

    Examples: init_params(4, 3, 2) -> 23; init_params(784, 128, 10) -> 101770.
    """
    return int(sum(np.asarray(v).size for v in params.values()))


# ---------------------------------------------------------------- Task 4
def forward(X: np.ndarray, params: dict, hidden: str = "relu") -> tuple[np.ndarray, dict]:
    """§4: Z1 = X @ W1 + b1; A1 = g(Z1); Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2).

    hidden is "relu" (default) or "sigmoid" (the by-hand example of §6 uses sigmoid).
    Raise ValueError if X.shape[1] != W1.shape[0] (the classic transposed-input bug).
    Returns (A2 of shape (n, n_out) with values in (0, 1), cache = {"Z1", "A1", "Z2", "A2"}).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != params["W1"].shape[0]:
        raise ValueError(f"X has shape {X.shape}; expected (n, {params['W1'].shape[0]})")
    g = relu if hidden == "relu" else sigmoid
    Z1 = X @ params["W1"] + params["b1"]
    A1 = g(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


# ---------------------------------------------------------------- Task 5
def bce_loss(y_hat: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """§5: -mean( y*log(y_hat + eps) + (1-y)*log(1 - y_hat + eps) ) over all entries.

    y_hat and y have the same shape ((n, 1) or (n,)). With y_hat == 0.5 everywhere the
    loss is ln 2 ≈ 0.693; a confident correct prediction gives ≈ 0; y_hat exactly 0 or 1
    must not produce inf/NaN.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float).reshape(y_hat.shape)
    return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))


# ---------------------------------------------------------------- Task 6
def backward(X: np.ndarray, Y: np.ndarray, params: dict, cache: dict, hidden: str = "relu") -> dict:
    """§6, the four equations, for BCE + sigmoid output and n = len(X):

    dZ2 = (A2 - Y) / n                     (1) output error
    dW2 = A1.T @ dZ2;  db2 = dZ2.sum(0)    (2)
    dZ1 = (dZ2 @ W2.T) * g'(Z1)            (3) push back through W2, gate by the derivative
    dW1 = X.T @ dZ1;   db1 = dZ1.sum(0)    (4)
    Returns {"dW1", "db1", "dW2", "db2"}, each with the SAME shape as its parameter.

    Check: X=[[1]], Y=[[1]], W1=[[0.5]], b1=[0], W2=[[-1]], b2=[0], hidden="sigmoid"
    gives dW2 = -0.4051, db2 = -0.6508, dW1 = db1 = 0.1529 (the worked example of §6).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(cache["A2"].shape)
    n = X.shape[0]
    g_prime = relu_grad if hidden == "relu" else sigmoid_grad
    dZ2 = (cache["A2"] - Y) / n
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ params["W2"].T) * g_prime(cache["Z1"])
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ---------------------------------------------------------------- Task 7
def train_mlp(X: np.ndarray, y: np.ndarray, n_hidden: int = 16, lr: float = 1.0,
              epochs: int = 1000, seed: int = 1) -> tuple[dict, list[float]]:
    """§7: full-batch gradient descent. Y = y.reshape(-1, 1); params = init_params(d, n_hidden, 1, seed);
    per epoch: forward -> bce_loss -> backward -> params[k] -= lr * grads["d" + k].

    Returns (params, history) where history[i] is the loss at epoch i (len == epochs).
    On make_churn_data() the loss falls from ≈ 0.7 to ≈ 0.1 and test accuracy is > 0.9.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(-1, 1)
    params = init_params(X.shape[1], n_hidden, 1, seed)
    history = []
    for _ in range(epochs):
        A2, cache = forward(X, params)
        history.append(bce_loss(A2, Y))
        grads = backward(X, Y, params, cache)
        for k in params:
            params[k] = params[k] - lr * grads["d" + k]
    return params, history


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    """Labels 0/1 as a 1-D int array: 1 where forward(X, params)[0] > 0.5."""
    A2, _ = forward(X, params)
    return (A2.ravel() > 0.5).astype(int)


if __name__ == "__main__":
    z = np.array([-2.0, 0.0, 2.0])
    print("sigmoid", sigmoid(z).round(3), "| sigmoid'", sigmoid_grad(z).round(3), "| relu", relu(z), "| relu'", relu_grad(z))
    params = init_params(4, 3, 2)
    print("4->3->2 parameters:", count_parameters(params), "| 784->128->10:", count_parameters(init_params(784, 128, 10)))
    X_tr, X_te, y_tr, y_te = make_churn_data()
    A2, cache = forward(X_tr, init_params(2, 16))
    print("forward shapes:", {k: v.shape for k, v in cache.items()}, "| loss at init:", round(bce_loss(A2, y_tr.reshape(-1, 1)), 3))
    # gradient check on a tiny batch
    p = init_params(2, 5, 1, seed=3)
    Xs, Ys = X_tr[:8], y_tr[:8].reshape(-1, 1)
    grads = backward(Xs, Ys, p, forward(Xs, p)[1])
    for k in ["W1", "b1", "W2", "b2"]:
        num = numerical_gradient(lambda q: bce_loss(forward(Xs, q)[0], Ys), p, k)
        print(f"gradient check {k}: max |analytic - numeric| = {np.abs(grads['d' + k] - num).max():.2e}")
    params, hist = train_mlp(X_tr, y_tr)
    print(f"training: loss {hist[0]:.3f} -> {hist[-1]:.3f}; test accuracy {(predict(X_te, params) == y_te).mean():.3f}")
