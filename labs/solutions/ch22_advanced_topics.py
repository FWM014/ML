"""
Lab 22 — Five problems, five toolkits (Chapter: 22 — Recommenders, Anomalies, Causality & RL)
=============================================================================================

THE PROBLEM
-----------
Streamline is a small video-on-demand service with a data team of one: you. This
quarter the roadmap has five items that are not "predict a label from a table":
(1) a "because you watched" shelf built from a sparse ratings matrix, (2) an offline
ranking evaluation so the product manager stops arguing about which shelf is better,
(3) a fraud queue: 30 analysts' hours a week to review the strangest payment events,
(4) a retention offer that costs money, so it must go only to customers it will actually
change, and (5) two exploration problems: which of four banners to show, and a warehouse
robot that must learn a route around a broken tile. All data are synthetic and seeded;
every task returns numbers you can defend.

TASKS (make the tests in labs/tests/test_ch22.py pass, one at a time)
---------------------------------------------------------------------
1. recommend_items(R, user, k, top_n)      → item-item adjusted-cosine recommender (Section 1)
2. mf_holdout_rmse(R, k, holdout_frac)     → truncated-SVD matrix factorization, RMSE on hidden cells (Section 2)
3. precision_at_k / ndcg_at_k              → ranking metrics matching hand-computed values (Section 2)
4. anomaly_precision_at_k(X, y, k)         → IsolationForest scores vs injected outliers (Section 4)
5. t_learner_uplift(...)                   → who to treat: two models, subtract, decile check (Section 6)
6. simulate_bandit / compare_bandits       → ε-greedy vs UCB cumulative regret over seeds (Section 7)
7. q_learning_gridworld()                  → tabular Q-learning; the greedy policy must reach the goal (Section 8)

STRETCH
-------
- Add Thompson sampling to the bandit comparison and check it beats both.
- Replace the T-learner with an X-learner and compare the top-decile observed uplift.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

USERS = ["Ana", "Ben", "Cy", "Dee", "Eli", "Fay"]
ITEMS = ["Dune", "Heat", "Up", "Alien", "Coco"]
RATINGS = np.array([[5, 4, 0, 5, 1],
                    [4, 0, 1, 4, 0],
                    [1, 1, 5, 0, 4],
                    [0, 1, 4, 1, 5],
                    [5, 5, 0, 4, 0],
                    [0, 2, 5, 1, 0]], dtype=float)         # 0 = not rated


def make_ratings_matrix(seed: int = 0, n_users: int = 80, n_items: int = 40, rank: int = 3,
                        density: float = 0.35) -> np.ndarray:
    """A larger synthetic ratings matrix with a true low-rank structure; 0 = missing. Do not modify."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_users, rank))
    V = rng.normal(0, 1, (n_items, rank))
    full = 3.0 + 0.8 * (U @ V.T) + rng.normal(0, 0.3, (n_users, n_items))
    full = np.clip(np.round(full), 1, 5)
    mask = rng.random((n_users, n_items)) < density
    return np.where(mask, full, 0.0)


def make_anomaly_data(seed: int = 0, n_normal: int = 1000, n_outliers: int = 30):
    """Two Gaussian blobs of normal payments plus uniformly scattered outliers. Returns (X, y). Do not modify."""
    from sklearn.datasets import make_blobs

    rng = np.random.default_rng(seed)
    X_norm, _ = make_blobs(n_samples=n_normal, centers=[[0, 0], [6, 5]], cluster_std=1.0, random_state=seed)
    X_out = rng.uniform(-6, 12, size=(n_outliers, 2))
    X = np.vstack([X_norm, X_out])
    y = np.r_[np.zeros(n_normal), np.ones(n_outliers)].astype(int)
    return X, y


def make_uplift_data(seed: int = 7, n: int = 12000):
    """A randomized retention campaign. Returns (X, T, Y, tau_true).

    x0 = price sensitivity, x1 = loyalty, x2/x3 = noise. Persuadables (x0 > 0.5) gain +0.25,
    sleeping dogs (x1 > 1.0) lose 0.10; loyal customers stay anyway. Do not modify.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    T = (rng.random(n) < 0.5).astype(int)
    base = 1 / (1 + np.exp(-0.8 * X[:, 1]))
    tau = 0.25 * (X[:, 0] > 0.5) - 0.10 * (X[:, 1] > 1.0)
    Y = (rng.random(n) < np.clip(base + tau * T, 0, 1)).astype(int)
    return X, T, Y, tau


# ---------------------------------------------------------------- Task 1
def recommend_items(R: np.ndarray, user: int, k: int = 2, top_n: int = 3) -> list[tuple[int, float]]:
    """Item-item collaborative filtering with adjusted cosine similarity.

    R is users x items, 0 = not rated. Steps:
      1. centre: C = R - (mean of each user's OWN ratings), with unrated cells set to 0
      2. similarity S[i, j] = cosine of C[:, i] and C[:, j] over users who rated BOTH items;
         0 if fewer than 2 such users (add 1e-9 to the denominator)
      3. predicted rating for an unrated item i: user_mean + sum(w * C[user, nbrs]) / sum(w), where
         nbrs are the k most similar items the user HAS rated with S > 0; if none, user_mean
      4. return [(item, predicted_rating), ...] for the user's UNRATED items only, highest first,
         at most top_n entries
    Example: recommend_items(RATINGS, user=1, k=2, top_n=1) -> [(1, 4.0)]   (Ben -> Heat 4.00)
    """
    # Approach: mask = R > 0; user means with np.nanmean; double loop for S; loop unrated items for predictions
    R = np.asarray(R, float)
    mask = R > 0
    n_items = R.shape[1]
    user_mean = np.nanmean(np.where(mask, R, np.nan), axis=1)
    C = np.where(mask, R - user_mean[:, None], 0.0)

    def sim(i, j):
        co = mask[:, i] & mask[:, j]
        if co.sum() < 2:
            return 0.0
        a, b = C[co, i], C[co, j]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    S = np.array([[sim(i, j) for j in range(n_items)] for i in range(n_items)])
    rated = np.where(mask[user])[0]
    out = []
    for i in range(n_items):
        if mask[user, i]:
            continue
        nbrs = rated[S[i, rated] > 0]
        nbrs = nbrs[np.argsort(-S[i, nbrs])[:k]]
        if len(nbrs) == 0:
            pred = float(user_mean[user])
        else:
            w = S[i, nbrs]
            pred = float(user_mean[user] + (w @ C[user, nbrs]) / w.sum())
        out.append((int(i), pred))
    out.sort(key=lambda t: -t[1])
    return out[:top_n]


# ---------------------------------------------------------------- Task 2
def mf_holdout_rmse(R: np.ndarray, k: int, holdout_frac: float = 0.2, seed: int = 0) -> dict:
    """Matrix factorization by truncated SVD, evaluated on hidden cells.

    1. Pick round(holdout_frac * n_known) KNOWN cells at random (rng = np.random.default_rng(seed),
       rng.choice without replacement over the indices of known cells) and hide them (set to 0).
    2. Fill every missing cell of the training matrix with the mean of its remaining known ratings.
    3. R_hat = rank-k reconstruction: U[:, :k] * s[:k] @ Vt[:k] from np.linalg.svd(full_matrices=False).
    4. Return {"R_hat": R_hat, "rmse_holdout": RMSE on the hidden cells, "rmse_train": RMSE on the
       remaining known cells}. With holdout_frac = 0 nothing is hidden and rmse_holdout is nan.
    """
    # Approach: known = np.argwhere(R > 0); rng.choice(len(known), n_hold, replace=False); np.linalg.svd
    R = np.asarray(R, float)
    rng = np.random.default_rng(seed)
    known = np.argwhere(R > 0)
    n_hold = int(round(holdout_frac * len(known)))
    hold_idx = rng.choice(len(known), n_hold, replace=False)
    hold = known[hold_idx]
    R_train = R.copy()
    R_train[hold[:, 0], hold[:, 1]] = 0.0
    train_mask = R_train > 0
    mu = R_train[train_mask].mean()
    filled = np.where(train_mask, R_train, mu)
    U, s, Vt = np.linalg.svd(filled, full_matrices=False)
    R_hat = (U[:, :k] * s[:k]) @ Vt[:k, :]
    rmse_hold = float(np.sqrt(np.mean((R_hat[hold[:, 0], hold[:, 1]] - R[hold[:, 0], hold[:, 1]]) ** 2))) if n_hold else float("nan")
    rmse_train = float(np.sqrt(np.mean((R_hat - R)[train_mask] ** 2)))
    return {"R_hat": R_hat, "rmse_holdout": rmse_hold, "rmse_train": rmse_train}


# ---------------------------------------------------------------- Task 3
def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    """Fraction of the top-k ranked items that are relevant.

    Example: ranked = ["Coco","Alien","Heat","Up","Dune"], relevant = {"Alien","Dune"}, k=3 -> 1/3
    """
    # Approach: len(set(ranked[:k]) & relevant) / k
    return len(set(ranked[:k]) & set(relevant)) / k


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance.

    DCG@k = sum over positions i (1-based) of rel_i / log2(i + 1); ideal DCG puts every relevant item
    first (min(len(relevant), k) of them). Return DCG / IDCG (0.0 if there are no relevant items).
    Example: same ranked/relevant as above, k=3 -> (1/log2(3)) / (1 + 1/log2(3)) = 0.3869
    """
    # Approach: gains list, dcg = sum(g / np.log2(i + 2)), ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    gains = [1.0 if it in relevant else 0.0 for it in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


# ---------------------------------------------------------------- Task 4
def anomaly_precision_at_k(X: np.ndarray, y_true: np.ndarray, k: int = 30, contamination: float = 0.03,
                           seed: int = 0) -> dict:
    """Score every row with an IsolationForest and evaluate the review queue.

    Fit IsolationForest(contamination=contamination, random_state=seed) on X (no labels!) and use
    ``-score_samples(X)`` so that higher = more anomalous. Return
      "scores":          array (n,), higher = stranger
      "top_k":           indices of the k highest scores (the analysts' queue), strangest first
      "precision_at_k":  fraction of those k that are true anomalies (y_true == 1)
      "pr_auc":          sklearn.metrics.average_precision_score(y_true, scores)
    """
    # Approach: IsolationForest(...).fit(X); scores = -det.score_samples(X); np.argsort(-scores)[:k]
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score

    det = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    scores = -det.score_samples(X)
    top_k = np.argsort(-scores)[:k]
    return {"scores": scores, "top_k": top_k, "precision_at_k": float(np.asarray(y_true)[top_k].mean()),
            "pr_auc": float(average_precision_score(y_true, scores))}


# ---------------------------------------------------------------- Task 5
def t_learner_uplift(X_tr, T_tr, Y_tr, X_te, T_te, Y_te, seed: int = 0) -> dict:
    """T-learner: one outcome model per arm, subtract, then validate by decile on a randomized holdout.

    Fit GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed) on the treated
    training rows (m1) and another on the control rows (m0). tau_hat = m1.predict_proba(X_te)[:, 1]
    - m0.predict_proba(X_te)[:, 1]. Sort the test rows by tau_hat (highest first), cut into 10 equal
    deciles (len // 10 rows each) and in each compute the OBSERVED uplift =
    mean(Y | T=1) - mean(Y | T=0). Return
      {"tau_hat": array, "decile_uplift": array (10,), "ate": mean(Y_te|T=1) - mean(Y_te|T=0)}
    """
    # Approach: boolean masks T_tr == 1 / == 0; np.argsort(-tau_hat); slice(i*d, (i+1)*d) per decile
    from sklearn.ensemble import GradientBoostingClassifier

    T_tr, Y_tr, T_te, Y_te = (np.asarray(a) for a in (T_tr, Y_tr, T_te, Y_te))
    m1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    m0 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
    tau_hat = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]
    order = np.argsort(-tau_hat)
    Yo, To = Y_te[order], T_te[order]
    d = len(order) // 10
    deciles = []
    for i in range(10):
        sl = slice(i * d, (i + 1) * d)
        deciles.append(float(Yo[sl][To[sl] == 1].mean() - Yo[sl][To[sl] == 0].mean()))
    ate = float(Y_te[T_te == 1].mean() - Y_te[T_te == 0].mean())
    return {"tau_hat": tau_hat, "decile_uplift": np.asarray(deciles), "ate": ate}


# ---------------------------------------------------------------- Task 6
def simulate_bandit(p_true: np.ndarray, policy: str, horizon: int = 3000, seed: int = 0,
                    eps: float = 0.1, c: float = 0.5) -> np.ndarray:
    """Run one bandit episode and return the CUMULATIVE expected regret after each step (shape (horizon,)).

    Arms are Bernoulli with click rates ``p_true``. rng = np.random.default_rng(seed). At step t = 1..horizon:
      "eps-greedy": with prob eps pick rng.integers(K), else argmax of the empirical mean s/max(n, 1)
      "ucb":        pull each arm once first (t <= K -> arm t-1), then argmax of s/n + sqrt(c * ln t / n)
      "uniform":    rng.integers(K)
    Draw the reward as rng.random() < p_true[a]; regret at step t is max(p_true) - p_true[a].
    (Use exactly one rng.random() for the eps coin, then rng.integers for the random arm, then one
    rng.random() for the reward, so that runs are reproducible.)
    """
    # Approach: n, s = zeros(K); loop t; choose a per policy; r = rng.random() < p_true[a]; accumulate regret
    p_true = np.asarray(p_true, float)
    K = len(p_true)
    best = p_true.max()
    rng = np.random.default_rng(seed)
    n = np.zeros(K)
    s = np.zeros(K)
    regret = np.zeros(horizon)
    total = 0.0
    for t in range(1, horizon + 1):
        if policy == "eps-greedy":
            a = int(rng.integers(K)) if rng.random() < eps else int(np.argmax(s / np.maximum(n, 1)))
        elif policy == "ucb":
            a = t - 1 if t <= K else int(np.argmax(s / n + np.sqrt(c * np.log(t) / n)))
        elif policy == "uniform":
            a = int(rng.integers(K))
        else:
            raise ValueError(f"unknown policy {policy!r}")
        r = rng.random() < p_true[a]
        n[a] += 1
        s[a] += r
        total += best - p_true[a]
        regret[t - 1] = total
    return regret


def compare_bandits(p_true: np.ndarray, horizon: int = 3000, seeds=range(8), policies=("eps-greedy", "ucb", "uniform")) -> dict:
    """Mean FINAL cumulative regret per policy, averaged over ``seeds``. Example: {"eps-greedy": 91.2, "ucb": 74.9, "uniform": 456.1}"""
    # Approach: dict comprehension over policies; np.mean of simulate_bandit(...)[-1] over seeds
    return {pol: float(np.mean([simulate_bandit(p_true, pol, horizon, seed=s)[-1] for s in seeds])) for pol in policies}


# ---------------------------------------------------------------- Task 7
def q_learning_gridworld(n_episodes: int = 500, seed: int = 1, alpha: float = 0.2, gamma: float = 0.9,
                         eps: float = 0.2, size: int = 4, goal=(3, 3), pit=(1, 2)) -> dict:
    """Tabular Q-learning on a size x size gridworld: start (0, 0), goal +1, pit -1, step cost -0.04.

    Actions 0..3 = up, right, down, left; moving into a wall keeps the agent in place. An episode ends
    on the goal or the pit or after 50 steps. Exploration: with prob eps a random action, else
    argmax Q[s]. Update: Q[s][a] += alpha * (target - Q[s][a]) with target = r if done else
    r + gamma * max Q[s2]. rng = np.random.default_rng(seed); draw rng.random() for the eps coin and
    rng.integers(4) for a random action.
    Return {"Q": array (size, size, 4), "policy": array (size, size) of argmax actions,
            "path": list of states visited by following the greedy policy from (0, 0), starting with
            (0, 0) and ending at the goal (or after size*size steps if it never gets there),
            "reached_goal": bool}
    """
    # Approach: moves dict; a step(s, a) helper; two nested loops for training; then a greedy walk
    rng = np.random.default_rng(seed)
    moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    Q = np.zeros((size, size, 4))

    def step(s, a):
        r, c = s
        dr, dc = moves[a]
        s2 = (min(max(r + dr, 0), size - 1), min(max(c + dc, 0), size - 1))
        if s2 == tuple(goal):
            return s2, 1.0, True
        if s2 == tuple(pit):
            return s2, -1.0, True
        return s2, -0.04, False

    for _ in range(n_episodes):
        s = (0, 0)
        for _ in range(50):
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = step(s, a)
            target = r if done else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                break
    policy = Q.argmax(axis=2)
    path = [(0, 0)]
    s = (0, 0)
    for _ in range(size * size):
        s, _, done = step(s, int(policy[s]))
        path.append(s)
        if done:
            break
    return {"Q": Q, "policy": policy, "path": path, "reached_goal": path[-1] == tuple(goal)}


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("[1] item-item recommendations:")
    for u in range(len(USERS)):
        recs = recommend_items(RATINGS, u, k=2, top_n=2)
        print(f"  {USERS[u]:<4}-> " + ", ".join(f"{ITEMS[i]} ({p:.2f})" for i, p in recs))

    print("\n[2] matrix factorization, RMSE on hidden cells:")
    R = make_ratings_matrix()
    for k in [1, 2, 3, 5, 10]:
        r = mf_holdout_rmse(R, k)
        print(f"  rank {k:>2}: holdout RMSE = {r['rmse_holdout']:.3f}   train RMSE = {r['rmse_train']:.3f}")

    print("\n[3] ranking metrics:")
    ranked, relevant = ["Coco", "Alien", "Heat", "Up", "Dune"], {"Alien", "Dune"}
    for k in (1, 3, 5):
        print(f"  k={k}: P@k={precision_at_k(ranked, relevant, k):.2f}  NDCG@k={ndcg_at_k(ranked, relevant, k):.2f}")

    print("\n[4] anomaly queue:")
    X, y = make_anomaly_data()
    a = anomaly_precision_at_k(X, y, k=30)
    print(f"  precision@30 = {a['precision_at_k']:.2f}   PR-AUC = {a['pr_auc']:.2f}")

    print("\n[5] uplift by decile (observed treated - control):")
    X, T, Y, tau = make_uplift_data()
    tr = np.arange(len(X)) < 7000
    u = t_learner_uplift(X[tr], T[tr], Y[tr], X[~tr], T[~tr], Y[~tr])
    print("  ", u["decile_uplift"].round(3), f"| ATE = {u['ate']:+.3f}")

    print("\n[6] bandits, mean final regret over 8 seeds:")
    print("  ", {k: round(v, 1) for k, v in compare_bandits([0.05, 0.12, 0.30, 0.18]).items()})

    print("\n[7] gridworld policy:")
    g = q_learning_gridworld()
    arrows = "↑→↓←"
    for r in range(4):
        print("  " + " ".join("G" if (r, c) == (3, 3) else "X" if (r, c) == (1, 2) else arrows[g["policy"][r, c]] for c in range(4)))
    print("  greedy path:", g["path"], "| reached goal:", g["reached_goal"])
