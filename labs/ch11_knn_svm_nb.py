"""
Lab 11 — Routing support tickets with the classics (Chapter: 11 — kNN, SVM & Naive Bayes)
==========================================================================================

THE PROBLEM
-----------
Kestrel Software's helpdesk receives ~600 tickets a week. Each has a short text and
a few structured fields: customer tier (1–3), number of prior tickets, hours the ticket
has been open, product code and message length. Tickets go either to Billing (0) or to
Technical (1), and today a dispatcher routes them by hand. You have last month's
tickets with the team that eventually resolved each one. The head of support wants a
router by Friday and has three questions: is "find the most similar past tickets" good
enough (and why does it fall apart if you forget to scale)? How much does a properly
tuned SVM add? And can the text alone route a ticket with a model simple enough to
explain (naive Bayes — which you will implement yourself to prove you understand the
smoothing)? Finish with the comparison loop that decides which model ships.

TASKS (make the tests in labs/tests/test_ch11.py pass, one at a time)
---------------------------------------------------------------------
1. distance_concentration(n, dims, seed)    → nearest/farthest distance ratio per dimension (the curse, in numbers)
2. knn_predict(X_tr, y_tr, X_q, k)          → kNN from scratch: Euclidean distance + majority vote, must match sklearn
3. scaling_effect(X, y, k)                  → CV accuracy of kNN with vs without a StandardScaler in the pipeline
4. tune_svm(X_tr, y_tr)                     → GridSearchCV over C and gamma for an RBF SVC; best params + the C x gamma table
5. naive_bayes_from_scratch(docs, labels, queries, alpha) → MultinomialNB with Laplace smoothing by hand, matching sklearn
6. compare_models(X, y, seed)               → LogReg / kNN / SVM / NB / RF on the SAME folds, sorted by ROC-AUC

STRETCH (no tests)
------------------
* Append 100 columns of standard-normal noise to the ticket features and re-run
  scaling_effect. Then put SelectKBest(f_classif, k=5) inside the pipeline. Explain the
  three numbers with the curse of dimensionality.
* Replace naive Bayes with TfidfVectorizer + LinearSVC on the ticket texts and compare
  macro-F1 with the same folds. Which would you ship if the dispatcher wants a
  confidence score next to every routed ticket?

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICKET_FEATURES = ["tier", "prior_tickets", "hours_open", "product_code", "message_len"]

TICKETS = [
    "invoice charged twice this month please refund", "cannot log in after password reset error 500",
    "update credit card on file for subscription", "app crashes on startup since latest update",
    "why was my plan upgraded without consent refund", "api returns timeout error on every request",
    "need a copy of last quarter invoices for accounting", "sync fails with error code 42 on mobile",
    "discount code not applied at checkout", "export to csv produces corrupted file",
    "cancel subscription and refund remaining balance", "login page blank after browser update",
    "billing address wrong on invoice", "integration webhook stopped sending events",
    "charged in wrong currency on my card", "dashboard charts not loading error in console",
]
TICKET_TEAMS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = billing, 1 = technical


def make_ticket_data(seed: int = 0, n: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    """Structured ticket fields and the resolving team (0 billing, 1 technical). Do not modify.

    The signal lives in tier, prior_tickets and product_code; hours_open is pure noise on
    a huge scale (so unscaled distances are dominated by it).
    """
    rng = np.random.default_rng(seed)
    tier = rng.integers(1, 4, n)
    prior = rng.poisson(2.0, n)
    hours = np.round(rng.lognormal(5.5, 0.9, n), 1)         # 50 .. 5000 hours, no signal
    product = rng.integers(1, 6, n)
    msg_len = np.clip(rng.normal(220, 70, n), 30, None).round(0)
    logit = 2.0 * (tier - 2) + 0.9 * (prior - 2) + 1.2 * (product >= 4) - 0.004 * (msg_len - 220)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"tier": tier, "prior_tickets": prior, "hours_open": hours,
                      "product_code": product, "message_len": msg_len})
    return X, y


# ---------------------------------------------------------------- Task 1
def distance_concentration(n: int = 1000, dims=(1, 2, 10, 100, 1000), seed: int = 0) -> pd.DataFrame:
    """Reproduce the chapter's curse-of-dimensionality table (Section 2).

    For each d in dims: draw X = rng.random((n, d)) and a query q = rng.random(d) from
    ONE np.random.default_rng(seed) (in that order, per d), compute the Euclidean
    distance from q to every row, and record nearest = min, farthest = max,
    ratio = nearest / farthest.
    Returns a DataFrame with columns ["d", "nearest", "farthest", "ratio"].

    Example: d=1 -> ratio ~0.00; d=100 -> ~0.70; d=1000 -> ~0.90.
    """
    # TODO: one rng for all dims; per d draw X then q; dist = sqrt(((X - q)**2).sum(axis=1))
    raise NotImplementedError("Task: distance_concentration")


# ---------------------------------------------------------------- Task 2
def knn_predict(X_tr, y_tr, X_q, k: int = 5) -> np.ndarray:
    """k-nearest neighbours from scratch (Section 1): Euclidean distance, majority vote.

    For every query row: compute its Euclidean distance to every training row, take the
    k closest (np.argsort, ties by position), and predict the most common label among
    them; if two labels tie, predict the SMALLER label (this is what sklearn does).
    Returns an int array of shape (len(X_q),). Uses no sklearn.

    Example: training points 0,1,2 -> label 0 and 10,11,12 -> label 1 (1-D);
             knn_predict(..., [[1.5], [11.0]], k=3) -> [0, 1]
    """
    # TODO: broadcast (X_q[:, None, :] - X_tr[None, :, :]) -> distances (n_q, n_tr); argsort; np.bincount vote
    raise NotImplementedError("Task: knn_predict")


# ---------------------------------------------------------------- Task 3
def scaling_effect(X, y, k: int = 5, seed: int = 0) -> dict[str, float]:
    """Cross-validated accuracy of kNN without and with scaling (Section 2, "Scaling is mandatory").

    cv = StratifiedKFold(5, shuffle=True, random_state=seed).
    "raw"    : cross_val_score(KNeighborsClassifier(k), X, y, cv=cv).mean()
    "scaled" : same, with make_pipeline(StandardScaler(), KNeighborsClassifier(k)) — the
               scaler is INSIDE the pipeline so it is fit on each training fold only.
    Returns {"raw": float, "scaled": float}.

    Example: on load_wine with k=5: raw ~0.66, scaled ~0.96.
    """
    # TODO: two cross_val_score calls with the same cv object; never call scaler.fit on the whole X
    raise NotImplementedError("Task: scaling_effect")


# ---------------------------------------------------------------- Task 4
def tune_svm(X_tr, y_tr, cv: int = 5):
    """GridSearchCV over C and gamma for Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="rbf"))]).

    Grid: {"svm__C": [0.1, 1, 10, 100], "svm__gamma": [0.001, 0.01, 0.1, 1]}, scoring="accuracy".
    Returns (best_params, table, search) where best_params = search.best_params_ and
    table is the cv_results_ pivoted to index=param_svm__C, columns=param_svm__gamma,
    values=mean_test_score (Section 4). Use n_jobs=1.

    Example: on load_breast_cancer (chapter split) best_params == {"svm__C": 100, "svm__gamma": 0.001}
             and the gamma=1 column is ~0.63 everywhere (every point became its own island).
    """
    # TODO: build the pipeline + grid, fit GridSearchCV, pivot pd.DataFrame(search.cv_results_)
    raise NotImplementedError("Task: tune_svm")


# ---------------------------------------------------------------- Task 5
def naive_bayes_from_scratch(docs: list[str], labels, queries: list[str], alpha: float = 1.0) -> np.ndarray:
    """Multinomial naive Bayes with Laplace smoothing, by hand (Section 5).

    Tokenise with CountVectorizer() fit on `docs` (so the vocabulary and tokens are the
    same as sklearn's). With counts C (n_docs x V) and binary labels:
      prior_c        = (#docs in class c) / n_docs
      P(w | c)       = (count(w, c) + alpha) / (N_c + alpha * V)      N_c = total tokens in class c
      log score_c(q) = log prior_c + sum_w count_q(w) * log P(w | c)
      P(1 | q)       = softmax over the two class scores (subtract the max before exp)
    Returns an array of P(class 1 | query) of shape (len(queries),), matching
    MultinomialNB(alpha=alpha).predict_proba(...)[:, 1] to 1e-6. Words not in the
    vocabulary are ignored (CountVectorizer.transform does this for you).

    Example: on the chapter's spam corpus P(spam | "free cash prize") = 0.94 and
             P(spam | "free report") = 0.41.
    """
    # TODO: vec = CountVectorizer().fit(docs); C = vec.transform(docs).toarray(); per-class token sums; logs; softmax
    raise NotImplementedError("Task: naive_bayes_from_scratch")


# ---------------------------------------------------------------- Task 6
def compare_models(X, y, seed: int = 1) -> pd.DataFrame:
    """The comparison loop (Section 7): five models, the SAME folds, ROC-AUC.

    cv = StratifiedKFold(5, shuffle=True, random_state=seed). Models (names exactly):
      "LogReg"       make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
      "kNN k=9"      make_pipeline(StandardScaler(), KNeighborsClassifier(9))
      "SVM RBF"      make_pipeline(StandardScaler(), SVC(C=10, gamma=0.01))
      "GaussNB"      GaussianNB()
      "RandomForest" RandomForestClassifier(200, random_state=seed)
    Returns a DataFrame with columns ["model", "auc_mean", "auc_std"], sorted by
    auc_mean descending, index reset.

    Example: on load_breast_cancer every model scores > 0.97 and the top two are within
             one standard deviation of each other — prefer the simpler one.
    """
    # TODO: build the dict of models, loop with cross_val_score(scoring="roc_auc"), collect, sort
    raise NotImplementedError("Task: compare_models")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    print(distance_concentration().round(2).to_string(index=False))

    X, y = make_ticket_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    sc = StandardScaler().fit(X_tr)
    mine = knn_predict(sc.transform(X_tr), y_tr, sc.transform(X_te), k=5)
    sk = KNeighborsClassifier(5).fit(sc.transform(X_tr), y_tr).predict(sc.transform(X_te))
    print(f"kNN from scratch: accuracy {(mine == y_te).mean():.3f}; agrees with sklearn on {(mine == sk).mean():.1%} of tickets")
    print("scaling effect on tickets:", {k: round(v, 3) for k, v in scaling_effect(X, y, k=5).items()})
    Xw, yw = load_wine(return_X_y=True)
    print("scaling effect on wine   :", {k: round(v, 3) for k, v in scaling_effect(Xw, yw, k=5).items()})

    Xb, yb = load_breast_cancer(return_X_y=True)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, random_state=0, stratify=yb)
    best, table, search = tune_svm(Xb_tr, yb_tr)
    print("SVM best params:", best, f"CV acc {search.best_score_:.3f}, test acc {search.score(Xb_te, yb_te):.3f}")
    print(table.round(3))

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    queries = ["refund the invoice please", "error after update", "invoice error"]
    mine = naive_bayes_from_scratch(TICKETS, TICKET_TEAMS, queries)
    sk = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(TICKETS, TICKET_TEAMS).predict_proba(queries)[:, 1]
    for q, a, b in zip(queries, mine, sk):
        print(f"{q!r:30s} P(technical) by hand = {a:.3f}  sklearn = {b:.3f}")

    print(compare_models(Xb, yb).round(3).to_string(index=False))
