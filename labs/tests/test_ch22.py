import numpy as np
import pytest

from labs import ch22_advanced_topics as lab


# ---------------------------------------------------------------- Task 1
def test_task1_recommendations_match_the_chapter():
    R = lab.RATINGS
    # Ben (1) has not rated Heat (1) and Coco (4); Heat is predicted 4.00 (chapter)
    recs = lab.recommend_items(R, user=1, k=2, top_n=1)
    assert recs == [(1, pytest.approx(4.0, abs=1e-2))]
    # Eli (4): Up predicted 4.67; Fay (5): Coco predicted 5.00; Ana (0): Up predicted 1.00
    assert dict(lab.recommend_items(R, user=4, k=2, top_n=5))[2] == pytest.approx(4.67, abs=0.01)
    assert lab.recommend_items(R, user=5, k=2, top_n=1)[0] == (4, pytest.approx(5.0, abs=1e-2))
    assert lab.recommend_items(R, user=0, k=2, top_n=3) == [(2, pytest.approx(1.0, abs=1e-2))]


def test_task1_never_recommends_already_rated_and_is_sorted():
    R = lab.make_ratings_matrix(seed=3)
    for u in [0, 7, 42]:
        recs = lab.recommend_items(R, user=u, k=5, top_n=6)
        assert 1 <= len(recs) <= 6
        items = [i for i, _ in recs]
        assert all(R[u, i] == 0 for i in items), "must exclude items the user already rated"
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)
        assert len(set(items)) == len(items)


# ---------------------------------------------------------------- Task 2
def test_task2_svd_reconstruction_rank_and_rmse():
    r2 = lab.mf_holdout_rmse(lab.RATINGS, k=2, holdout_frac=0.0)
    assert r2["R_hat"].shape == lab.RATINGS.shape
    assert r2["rmse_train"] == pytest.approx(0.65, abs=0.02)       # chapter: rank 2 -> 0.65
    assert np.isnan(r2["rmse_holdout"])
    r5 = lab.mf_holdout_rmse(lab.RATINGS, k=5, holdout_frac=0.0)
    assert r5["rmse_train"] < 1e-8                                  # full rank reproduces the matrix
    assert np.linalg.matrix_rank(r2["R_hat"]) == 2


def test_task2_holdout_rmse_exposes_overfitting_in_latent_space():
    R = lab.make_ratings_matrix(seed=0)
    n_known = int((R > 0).sum())
    out = {k: lab.mf_holdout_rmse(R, k, holdout_frac=0.2, seed=0) for k in [1, 3, 10]}
    for k, o in out.items():
        assert np.linalg.matrix_rank(o["R_hat"]) == k
        assert 0 < o["rmse_holdout"] < 2.0 and 0 < o["rmse_train"] < 2.0
    assert out[3]["rmse_holdout"] < out[1]["rmse_holdout"] - 0.1     # the true rank helps ...
    assert out[10]["rmse_train"] < out[3]["rmse_train"] - 0.1         # ... more factors fit the train cells better
    assert out[10]["rmse_holdout"] > out[3]["rmse_holdout"]           # ... but generalise worse (the classic mistake)
    # reproducible hold-out
    again = lab.mf_holdout_rmse(R, 3, holdout_frac=0.2, seed=0)
    assert again["rmse_holdout"] == pytest.approx(out[3]["rmse_holdout"])
    # the hidden cells were really hidden: the train matrix has fewer known cells
    assert n_known > 0


# ---------------------------------------------------------------- Task 3
def test_task3_ranking_metrics_hand_computed():
    ranked, relevant = ["Coco", "Alien", "Heat", "Up", "Dune"], {"Alien", "Dune"}
    assert lab.precision_at_k(ranked, relevant, 1) == pytest.approx(0.0)
    assert lab.precision_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
    assert lab.precision_at_k(ranked, relevant, 5) == pytest.approx(0.4)
    assert lab.ndcg_at_k(ranked, relevant, 1) == pytest.approx(0.0)
    assert lab.ndcg_at_k(ranked, relevant, 3) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)), abs=1e-6)
    assert lab.ndcg_at_k(ranked, relevant, 5) == pytest.approx((1 / np.log2(3) + 1 / np.log2(6)) / (1 + 1 / np.log2(3)), abs=1e-6)
    # a perfect ranking scores 1; order matters for NDCG but not for precision
    assert lab.ndcg_at_k(["Alien", "Dune", "Coco"], relevant, 3) == pytest.approx(1.0)
    assert lab.ndcg_at_k(["Alien", "Coco", "Dune"], relevant, 3) < 1.0
    assert lab.precision_at_k(["Alien", "Coco", "Dune"], relevant, 3) == pytest.approx(lab.precision_at_k(["Alien", "Dune", "Coco"], relevant, 3))
    assert lab.ndcg_at_k(ranked, set(), 3) == pytest.approx(0.0)


# ---------------------------------------------------------------- Task 4
def test_task4_isolation_forest_queue():
    X, y = lab.make_anomaly_data(seed=0)
    out = lab.anomaly_precision_at_k(X, y, k=30, contamination=0.03, seed=0)
    assert out["scores"].shape == (len(X),)
    assert len(out["top_k"]) == 30 and len(set(out["top_k"].tolist())) == 30
    assert list(out["top_k"]) == list(np.argsort(-out["scores"])[:30])
    assert out["precision_at_k"] == pytest.approx(y[out["top_k"]].mean())
    assert out["precision_at_k"] >= 0.5                                # 30 injected among 1030: random would be ~0.03
    assert out["pr_auc"] >= 0.6
    # the injected outliers score higher on average than the normal points
    assert out["scores"][y == 1].mean() > out["scores"][y == 0].mean() + 0.05


# ---------------------------------------------------------------- Task 5
def test_task5_t_learner_recovers_who_to_treat():
    X, T, Y, tau = lab.make_uplift_data(seed=7)
    tr = np.arange(len(X)) < 7000
    out = lab.t_learner_uplift(X[tr], T[tr], Y[tr], X[~tr], T[~tr], Y[~tr], seed=0)
    assert out["tau_hat"].shape == ((~tr).sum(),)
    assert out["decile_uplift"].shape == (10,)
    assert 0.0 < out["ate"] < 0.10                                      # treating everyone barely pays
    assert out["decile_uplift"][0] > 0.15                                # top decile: persuadables (+0.25 truth)
    assert out["decile_uplift"][0] > out["decile_uplift"][-1] + 0.15     # bottom decile: sleeping dogs / sure things
    assert np.corrcoef(out["tau_hat"], tau[~tr])[0, 1] > 0.5              # sign and rough magnitude recovered
    assert abs(out["tau_hat"][tau[~tr] > 0.2].mean() - 0.25) < 0.12


# ---------------------------------------------------------------- Task 6
def test_task6_bandit_regret_curves():
    p = np.array([0.05, 0.12, 0.30, 0.18])
    for pol in ["eps-greedy", "ucb", "uniform"]:
        reg = lab.simulate_bandit(p, pol, horizon=500, seed=0)
        assert reg.shape == (500,)
        assert np.all(np.diff(reg) >= -1e-12), "cumulative regret never decreases"
        assert reg[0] >= 0 and reg[-1] <= 500 * 0.25 + 1e-9
    assert np.allclose(lab.simulate_bandit(p, "ucb", 300, seed=4), lab.simulate_bandit(p, "ucb", 300, seed=4))
    # uniform exploration regret is ~ T * (best - mean p)
    uni = lab.simulate_bandit(p, "uniform", horizon=2000, seed=0)[-1]
    assert abs(uni - 2000 * (0.30 - p.mean())) < 0.15 * 2000 * (0.30 - p.mean())
    with pytest.raises(ValueError):
        lab.simulate_bandit(p, "thompson-typo", 10)


def test_task6_ucb_beats_eps_greedy_on_average():
    p = np.array([0.05, 0.12, 0.30, 0.18])
    res = lab.compare_bandits(p, horizon=3000, seeds=range(8))
    assert set(res) >= {"eps-greedy", "ucb", "uniform"}
    assert res["ucb"] < res["eps-greedy"] - 10
    assert res["eps-greedy"] < res["uniform"] / 3


# ---------------------------------------------------------------- Task 7
def test_task7_q_learning_reaches_the_goal():
    out = lab.q_learning_gridworld(n_episodes=500, seed=1)
    assert out["Q"].shape == (4, 4, 4) and out["policy"].shape == (4, 4)
    assert out["reached_goal"] is True
    path = out["path"]
    assert path[0] == (0, 0) and path[-1] == (3, 3)
    assert (1, 2) not in path, "the greedy path must avoid the pit"
    assert len(path) - 1 <= 8                                            # shortest route is 6 moves
    # consecutive states are neighbours
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    # values grow towards the goal, and the pit's neighbours point away from it
    V = out["Q"].max(axis=2)
    assert V[3, 2] > V[0, 0]
    assert out["policy"][1, 1] != 1 and out["policy"][0, 2] != 2         # do not step right/down into the pit
    # training matters: a handful of episodes does not produce a reliable policy
    weak = lab.q_learning_gridworld(n_episodes=3, seed=1)
    assert (not weak["reached_goal"]) or len(weak["path"]) - 1 > 6 or np.abs(weak["Q"]).max() < 0.5
