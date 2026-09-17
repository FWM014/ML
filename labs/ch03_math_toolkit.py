"""
Lab 03 — The math toolkit, in numpy (Chapter 3: The Math You Actually Need)
===========================================================================
Chapter: 03 — The Math You Actually Need

THE PROBLEM
-----------
You have joined the data team at Nordwind, an online retailer with 40,000 products and
a small fraud problem. In one week you get four requests that are all "just math":
(1) Search wants "similar products" from the 8-number embedding each product already
has; (2) Marketing wants every customer scored against three campaign models at once,
and their last attempt crashed with a shape error; (3) Finance is optimising a
pricing loss and is not sure their hand-written gradient is right; (4) Risk runs a
fraud screen that flags 0.5% of orders and wants to know how many flags are real
before they hire reviewers; and (5) a colleague building a churn decision tree needs
to know whether a candidate split on `days_since_login` actually carries information.
Everything here is plain numpy: no scikit-learn, no scipy. Shapes are the contract.

TASKS (make the tests in labs/tests/test_ch03.py pass, one at a time)
---------------------------------------------------------------------
1. cosine_similarity(u, v)                → direction-only similarity in [-1, 1]
2. batch_predict(X, W, b)                 → (n,d) @ (d,k) + (k,) with helpful shape errors
3. bayes_posterior(prior, sens, spec)     → P(condition | positive test) and counts per 1,000
4. entropy(p)                             → expected surprise in bits
5. information_gain(y, mask)              → how much a candidate split reduces entropy
6. numerical_gradient(f, w, h)            → central finite differences, one knob at a time
7. gradient_descent(f, grad, w0, lr, n)   → gradient check, then walk downhill; losses fall

STRETCH (no tests)
------------------
- Implement softmax with the max-subtraction trick and show that np.exp overflows without it.
- Bootstrap a 95% interval for the mean of a skewed spend column (Section 8) and compare it
  with the s/sqrt(n) formula.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np


def make_product_embeddings(seed: int = 0, n: int = 200, d: int = 8) -> np.ndarray:
    """n product embeddings of dimension d, in three loose clusters. Do not modify."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, d)) * 3
    labels = rng.integers(0, 3, n)
    return centres[labels] + rng.normal(size=(n, d))


def pricing_loss(w: np.ndarray) -> float:
    """Finance's loss: a stretched bowl with its minimum at (2, -1). Do not modify."""
    return float((w[0] - 2) ** 2 + 3 * (w[1] + 1) ** 2)


def pricing_grad(w: np.ndarray) -> np.ndarray:
    """The analytic gradient of pricing_loss. Do not modify."""
    return np.array([2 * (w[0] - 2), 6 * (w[1] + 1)])


# ---------------------------------------------------------------- Task 1
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """cos(theta) = u.v / (|u| |v|)  (chapter §1 and §3).

    Length does not matter, only direction: cosine([1, 2], [10, 20]) == 1.0.

    Example: cosine_similarity([1, 0], [0, 1]) -> 0.0 ; cosine_similarity([1, 1], [-1, -1]) -> -1.0
    """
    # TODO: np.dot(u, v) divided by np.linalg.norm(u) * np.linalg.norm(v); return a float
    raise NotImplementedError("Task: cosine_similarity")


# ---------------------------------------------------------------- Task 2
def batch_predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Score every row against k linear models at once: X @ W + b  (chapter §2).

    Shapes: X is (n, d), W is (d, k), b is (k,). Result is (n, k); entry [i, j] is the dot
    product of row i of X with column j of W, plus b[j].
    Raise ValueError with a message that contains both offending shapes when
    X.shape[1] != W.shape[0] or b.shape != (W.shape[1],). Never "fix" a mismatch by transposing.

    Example: X = [[1, 2]], W = [[1, 0], [0, 1]], b = [10, 20] -> [[11, 22]]
    """
    # TODO: check X.shape[1] == W.shape[0] and b.shape == (W.shape[1],) and raise ValueError with the shapes; then return X @ W + b
    raise NotImplementedError("Task: batch_predict")


# ---------------------------------------------------------------- Task 3
def bayes_posterior(prior: float, sensitivity: float, specificity: float) -> dict:
    """Bayes' theorem for a screening test (chapter §6).

    prior = P(condition), sensitivity = P(positive | condition),
    specificity = P(negative | no condition).
    Returns {"p_positive": P(positive), "posterior": P(condition | positive),
             "per_1000": {"true_positives": ..., "false_positives": ...}}  (floats, per 1,000 cases).

    Example: prior=0.01, sensitivity=0.99, specificity=0.95 -> p_positive 0.0594,
             posterior 0.1667, per_1000 = {true_positives: 9.9, false_positives: 49.5}
    """
    # TODO: p_positive = prior*sens + (1-prior)*(1-spec); posterior = prior*sens / p_positive; per-1000 counts multiply by 1000
    raise NotImplementedError("Task: bayes_posterior")


# ---------------------------------------------------------------- Task 4
def entropy(p) -> float:
    """Entropy in bits of a probability vector p (chapter §9): -sum(p_i log2 p_i).

    0 * log 0 is defined as 0, so zero entries are skipped. Return at least 0.0.

    Example: entropy([0.5, 0.5]) -> 1.0 ; entropy([1.0]) -> 0.0 ; entropy([1/8]*8) -> 3.0
    """
    # TODO: drop zero entries (p[p > 0]); return max(0.0, -(p * np.log2(p)).sum())
    raise NotImplementedError("Task: entropy")


# ---------------------------------------------------------------- Task 5
def information_gain(y: np.ndarray, mask: np.ndarray) -> float:
    """Information gain of splitting the binary labels y by a boolean mask (chapter §9).

    gain = H(y) - [ n_left/n * H(y[mask]) + n_right/n * H(y[~mask]) ]
    where H is the entropy of the class proportions in each group (use np.bincount / mean).
    An empty side contributes 0. The result is in bits and is never negative.

    Example: y = [1, 1, 0, 0], mask = [T, T, F, F] -> 1.0 (a perfect split of a fair node)
             y = [1, 0, 1, 0], mask = [T, T, F, F] -> 0.0 (the split told us nothing)
    """
    # TODO: entropy of np.bincount(labels)/len(labels) for the parent and each side; weight the children by their share of rows
    raise NotImplementedError("Task: information_gain")


# ---------------------------------------------------------------- Task 6
def numerical_gradient(f, w: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central finite differences (chapter §4): nudge each parameter by +h and -h.

    g[i] = (f(w + h e_i) - f(w - h e_i)) / (2h), where e_i is the unit vector for knob i.
    Returns an array with the same shape as w. Never modifies w.

    Example: f(w) = w[0]**2 + 3*w[1]**2 at w = [1, 1] -> [2.0, 6.0]
    """
    # TODO: loop over i; build a unit vector e with e[i] = h; g[i] = (f(w + e) - f(w - e)) / (2h)
    raise NotImplementedError("Task: numerical_gradient")


# ---------------------------------------------------------------- Task 7
def gradient_descent(f, grad, w0: np.ndarray, lr: float = 0.1, n_steps: int = 50,
                     check_tol: float = 1e-4) -> tuple[np.ndarray, list[float]]:
    """Check the gradient, then walk downhill: w <- w - lr * grad(w)  (chapter §4).

    Before the first step compare grad(w0) with numerical_gradient(f, w0); if any entry
    differs by more than check_tol (absolute), raise ValueError("gradient check failed").
    Then take n_steps steps and record f(w) AFTER each step.
    Returns (w_final, losses) with len(losses) == n_steps. Does not modify w0.

    Example: gradient_descent(pricing_loss, pricing_grad, [-1, 2], lr=0.1, n_steps=25)
             -> w close to [2, -1], losses decreasing from ~10 to ~0.0001
    """
    # TODO: compare grad(w0) with numerical_gradient(f, w0) first; then loop: w = w - lr * grad(w); losses.append(f(w))
    raise NotImplementedError("Task: gradient_descent")


if __name__ == "__main__":
    E = make_product_embeddings()
    sims = np.array([cosine_similarity(E[0], e) for e in E])
    top = np.argsort(-sims)[1:4]
    print(f"products most similar to #0: {top.tolist()} (cosines {sims[top].round(3).tolist()})")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 8)); W = rng.normal(size=(8, 3)); b = np.array([0.1, -0.2, 0.0])
    print("batch scores shape:", batch_predict(X, W, b).shape)
    try:
        batch_predict(X, W.T, b)
    except ValueError as e:
        print("shape error caught:", e)

    post = bayes_posterior(prior=0.005, sensitivity=0.90, specificity=0.99)
    print(f"fraud screen: P(fraud | flagged) = {post['posterior']:.3f}; per 1,000 orders "
          f"{post['per_1000']['true_positives']:.1f} true vs {post['per_1000']['false_positives']:.1f} false flags")

    print(f"entropy of a 26/74 churn node: {entropy([0.26, 0.74]):.3f} bits")
    y = (rng.random(400) < 0.26).astype(int)
    days = rng.integers(0, 90, 400)
    y = np.where(days > 60, (rng.random(400) < 0.5).astype(int), y)
    print(f"gain of splitting at days_since_login > 60: {information_gain(y, days > 60):.3f} bits; "
          f"at a random coin flip: {information_gain(y, rng.random(400) < 0.5):.4f} bits")

    w0 = np.array([-1.0, 2.0])
    print("numerical grad:", numerical_gradient(pricing_loss, w0).round(4), "analytic:", pricing_grad(w0))
    w, losses = gradient_descent(pricing_loss, pricing_grad, w0, lr=0.1, n_steps=25)
    print(f"after 25 steps: w = {w.round(3)}, loss {losses[0]:.3f} -> {losses[-1]:.5f}")
