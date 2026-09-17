import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from labs import ch11_knn_svm_nb as lab

SPAM_DOCS = ["win a free prize now", "free money click here", "claim a free reward today",
             "urgent offer win cash prize", "meeting moved to 3pm", "please review the attached report",
             "lunch tomorrow with the team", "quarterly budget review meeting",
             "can you send the report", "free lunch in the meeting room today"]
SPAM_LABELS = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_task1_distances_concentrate_as_dimensions_grow():
    tbl = lab.distance_concentration(n=1000, dims=(1, 2, 10, 100, 1000), seed=0)
    assert list(tbl.columns) == ["d", "nearest", "farthest", "ratio"]
    assert tbl["d"].tolist() == [1, 2, 10, 100, 1000]
    assert tbl["ratio"].is_monotonic_increasing
    assert tbl["ratio"].iloc[0] < 0.05
    assert tbl["ratio"].iloc[-1] > 0.85
    assert np.allclose(tbl["ratio"], tbl["nearest"] / tbl["farthest"])
    # the chapter's numbers (same rng recipe)
    assert tbl.loc[tbl["d"] == 100, "nearest"].item() == pytest.approx(3.34, abs=0.05)
    assert tbl.loc[tbl["d"] == 1000, "farthest"].item() == pytest.approx(13.55, abs=0.05)


def test_task2_knn_toy_and_tie_rule():
    X_tr = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y_tr = np.array([0, 0, 0, 1, 1, 1])
    out = lab.knn_predict(X_tr, y_tr, np.array([[1.5], [11.0], [5.9], [6.1]]), k=3)
    assert isinstance(out, np.ndarray) and out.dtype.kind == "i"
    assert out.tolist() == [0, 1, 0, 1]
    # even k with a 1-1 vote: the smaller label wins (sklearn's rule)
    out = lab.knn_predict(np.array([[0.0], [2.0]]), np.array([1, 0]), np.array([[1.0]]), k=2)
    assert out.tolist() == [0]


def test_task2_knn_matches_sklearn_on_tickets():
    X, y = lab.make_ticket_data(seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    sc = StandardScaler().fit(X_tr)
    A, B = sc.transform(X_tr), sc.transform(X_te)
    for k in (1, 5, 9):
        mine = lab.knn_predict(A, y_tr, B, k=k)
        sk = KNeighborsClassifier(n_neighbors=k).fit(A, y_tr).predict(B)
        assert mine.shape == sk.shape
        assert (mine == sk).mean() > 0.99  # identical up to distance ties
    assert (lab.knn_predict(A, y_tr, B, k=5) == y_te).mean() > 0.65


def test_task3_scaling_is_the_biggest_lever():
    X, y = load_wine(return_X_y=True)
    res = lab.scaling_effect(X, y, k=5, seed=0)
    assert set(res) == {"raw", "scaled"}
    assert res["raw"] < 0.75
    assert res["scaled"] > 0.93
    assert res["scaled"] > res["raw"] + 0.2
    # exact agreement with the in-pipeline computation (scaler fit per fold)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    ref = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=cv).mean()
    assert res["scaled"] == pytest.approx(ref, abs=1e-9)


def test_task3_scaler_is_fit_inside_the_folds():
    # the classic mistake: StandardScaler().fit_transform(X) on the whole table before CV gives a
    # different number on the ticket data; the honest per-fold pipeline must be reproduced exactly
    X, y = lab.make_ticket_data(seed=0)
    res = lab.scaling_effect(X, y, k=5, seed=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    honest = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=cv).mean()
    leaky = cross_val_score(KNeighborsClassifier(5), StandardScaler().fit_transform(X), y, cv=cv).mean()
    assert honest != pytest.approx(leaky, abs=1e-9)  # sanity: the two really differ here
    assert res["scaled"] == pytest.approx(honest, abs=1e-9)
    assert res["scaled"] > res["raw"] + 0.1


def test_task4_svm_grid_finds_the_chapter_optimum():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    best, table, search = lab.tune_svm(X_tr, y_tr, cv=5)
    assert best == {"svm__C": 100, "svm__gamma": 0.001}
    assert search.best_score_ > 0.98
    assert search.score(X_te, y_te) > 0.94
    assert table.shape == (4, 4)
    assert list(table.index) == [0.1, 1, 10, 100] and list(table.columns) == [0.001, 0.01, 0.1, 1]
    assert (table[1] < 0.7).all()          # gamma=1: every point is its own island
    assert table.loc[0.1, 0.001] < 0.8     # small C, small gamma: underfits
    assert table.loc[100, 0.001] == pytest.approx(search.best_score_)


def test_task5_naive_bayes_matches_sklearn_and_the_chapter():
    queries = ["free cash prize", "review the meeting report", "free report"]
    mine = lab.naive_bayes_from_scratch(SPAM_DOCS, SPAM_LABELS, queries, alpha=1.0)
    sk = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(SPAM_DOCS, SPAM_LABELS)
    assert mine.shape == (3,)
    assert np.allclose(mine, sk.predict_proba(queries)[:, 1], atol=1e-6)
    assert mine.tolist() == pytest.approx([0.94, 0.01, 0.41], abs=0.01)
    # a different smoothing strength, on the ticket corpus, also matches
    q2 = ["refund the invoice please", "error after update", "invoice error", "unknownword"]
    mine2 = lab.naive_bayes_from_scratch(lab.TICKETS, lab.TICKET_TEAMS, q2, alpha=0.3)
    sk2 = make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.3)).fit(lab.TICKETS, lab.TICKET_TEAMS)
    assert np.allclose(mine2, sk2.predict_proba(q2)[:, 1], atol=1e-6)
    assert mine2[0] < 0.2 and mine2[1] > 0.8
    assert mine2[3] == pytest.approx(0.5)  # no known words: only the (50/50) prior speaks


def test_task5_smoothing_prevents_the_zero_kill():
    # "report" never appears in spam: with alpha -> 0 the spam score collapses to ~0
    tiny = lab.naive_bayes_from_scratch(SPAM_DOCS, SPAM_LABELS, ["free report"], alpha=1e-9)
    smoothed = lab.naive_bayes_from_scratch(SPAM_DOCS, SPAM_LABELS, ["free report"], alpha=1.0)
    assert tiny[0] < 1e-6
    assert 0.3 < smoothed[0] < 0.5


def test_task6_comparison_loop_uses_the_same_folds():
    X, y = load_breast_cancer(return_X_y=True)
    tbl = lab.compare_models(X, y, seed=1)
    assert list(tbl.columns) == ["model", "auc_mean", "auc_std"]
    assert set(tbl["model"]) == {"LogReg", "kNN k=9", "SVM RBF", "GaussNB", "RandomForest"}
    assert tbl["auc_mean"].is_monotonic_decreasing
    assert list(tbl.index) == list(range(5))
    assert (tbl["auc_mean"] > 0.97).all()
    assert (tbl["auc_std"] < 0.03).all()
    assert tbl["model"].iloc[0] in {"SVM RBF", "LogReg"}
    # reproducible: identical folds and seeds give identical numbers
    tbl2 = lab.compare_models(X, y, seed=1)
    assert np.allclose(tbl["auc_mean"], tbl2["auc_mean"])
