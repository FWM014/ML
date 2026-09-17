"""
Lab 09 — Decision trees you can defend (Chapter: 09 — Decision Trees)
=====================================================================

THE PROBLEM
-----------
Meridian Credit Union has 1,500 personal loans that went 30+ days late last year, and
a collections team of six people who can call about 200 accounts a week. Which ones
first? Compliance has ruled that any prioritisation rule must fit on ONE page and be
applied by staff without software — so the deliverable is a shallow decision tree, not
a score. You have the loan table at the moment of first delinquency: late payments in
the previous 24 months, card utilisation, months since the oldest account, debt-to-
income, income, employment years — plus a column called `random_id` that the data
warehouse added (a uniform random number that looks suspiciously important in the
first tree somebody fitted). The label is whether the loan defaulted within 12 months.

Before you print the rules you must show you understand what the tree is doing: score
a split by hand, reproduce scikit-learn's root split from scratch, choose depth and
pruning strength by cross-validation, and prove that `random_id` is noise.

TASKS (make the tests in labs/tests/test_ch09.py pass, one at a time)
---------------------------------------------------------------------
1. impurity(counts, criterion)                  → Gini or entropy of a node's class counts
2. split_candidates(x, y)                       → every midpoint threshold with weighted Gini and information gain
3. best_split(X, y)                             → the (feature, threshold) a CART tree picks at the root — must match sklearn
4. best_depth(X, y, depths)                     → depth sweep with cross_val_score; the depth that generalises best
5. prune_alpha(X_tr, y_tr)                      → cost-complexity path + CV → best ccp_alpha and the pruned tree
6. importance_report(...)                       → impurity vs permutation importance, with noise features flagged
7. segment_table(X, y, feature_names, n_segments) → max_leaf_nodes tree turned into a rule/size/default-rate table

STRETCH (no tests)
------------------
* Draw 20 bootstrap samples, refit the depth-3 tree on each and count the distinct root
  features. Then raise min_samples_leaf to 30 and repeat. Would you hand the collections
  team a flowchart that changes next quarter?
* Fit DecisionTreeClassifier(max_depth=3) to the ERRORS of your pruned tree (1 where it
  was wrong). Which three rules describe the loans the model gets wrong?

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURES = ["late_payments_24m", "utilization", "months_oldest_account", "dti", "income",
            "employment_years", "random_id"]


def make_collections_data(seed: int = 0, n: int = 1500) -> tuple[pd.DataFrame, np.ndarray]:
    """Delinquent-loan table (DataFrame with FEATURES) and default label. Do not modify."""
    rng = np.random.default_rng(seed)
    late = rng.poisson(0.8, n)
    util = rng.beta(2, 3, n)
    oldest = rng.gamma(4, 30, n)
    dti = np.clip(rng.normal(0.3, 0.12, n), 0.02, 0.9)
    income = rng.lognormal(10.6, 0.45, n)
    employ = rng.gamma(2, 3, n)
    logit = (-1.4 + 0.9 * late + 2.5 * (util - 0.4) - 0.006 * (oldest - 120)
             + 3.0 * (dti - 0.3) - 0.5 * (np.log(income) - 10.6) - 0.05 * employ)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({
        "late_payments_24m": late,
        "utilization": util.round(3),
        "months_oldest_account": oldest.round(0),
        "dti": dti.round(3),
        "income": income.round(0),
        "employment_years": employ.round(1),
        "random_id": rng.random(n),          # pure noise, 1500 distinct values
    })
    return X, y


# The chapter's hand-worked example (Section 3): ten customers, support calls, churn label.
CALLS = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6])
CHURNED = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])


# ---------------------------------------------------------------- Task 1
def impurity(counts, criterion: str = "gini") -> float:
    """Impurity of a node from its class counts (Chapter 9, Section 3).

    gini    = 1 - sum(p_k^2);  entropy = -sum(p_k * log2(p_k)) over classes with p_k > 0.
    `counts` is a sequence of non-negative integers (one per class).

    Example: impurity([5, 5]) -> 0.5;  impurity([5, 5], "entropy") -> 1.0;
             impurity([8, 2]) -> 0.32;  impurity([4, 0], "entropy") -> 0.0
    """
    # TODO: p = counts / sum; gini = 1 - sum(p**2); entropy = -sum(p*log2 p) over p > 0; raise ValueError otherwise
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    if criterion == "gini":
        return float(1.0 - np.sum(p ** 2))
    if criterion == "entropy":
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
    raise ValueError(f"unknown criterion {criterion!r}")


# ---------------------------------------------------------------- Task 2
def split_candidates(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Score every candidate threshold of ONE numeric feature for a binary label.

    Candidates are the midpoints between consecutive distinct sorted values of x.
    For each threshold t: left = rows with x <= t, right = rows with x > t;
      weighted_gini = (n_L/n) * gini(left) + (n_R/n) * gini(right)
      info_gain     = entropy(parent) - [(n_L/n) * H(left) + (n_R/n) * H(right)]
    Returns a DataFrame with columns ["threshold", "weighted_gini", "info_gain"],
    one row per candidate, in increasing threshold order.

    Example (the chapter's table): split_candidates(CALLS, CHURNED) has thresholds
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]; at 1.5 weighted_gini = 0.267 and info_gain = 0.449.
    """
    # TODO: sort the distinct values, take midpoints, and for each threshold count (stay, churn) on each side
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    n = len(y)
    classes = np.unique(y)
    parent = [int((y == c).sum()) for c in classes]
    h_parent = impurity(parent, "entropy")
    vals = np.unique(x)
    rows = []
    for t in (vals[:-1] + vals[1:]) / 2:
        left, right = y[x <= t], y[x > t]
        cl = [int((left == c).sum()) for c in classes]
        cr = [int((right == c).sum()) for c in classes]
        wl, wr = len(left) / n, len(right) / n
        rows.append({
            "threshold": float(t),
            "weighted_gini": wl * impurity(cl) + wr * impurity(cr),
            "info_gain": h_parent - (wl * impurity(cl, "entropy") + wr * impurity(cr, "entropy")),
        })
    return pd.DataFrame(rows, columns=["threshold", "weighted_gini", "info_gain"])


# ---------------------------------------------------------------- Task 3
def best_split(X, y) -> tuple[int, float, float]:
    """The greedy root split (Section 5): scan every feature and every midpoint threshold.

    Returns (feature_index, threshold, weighted_gini) of the split with the LOWEST
    weighted Gini; on exact ties keep the first feature / lowest threshold found.
    X is a 2-D array or DataFrame (n rows, d features); y a binary label.

    Example: on make_collections_data() the answer must agree with
             DecisionTreeClassifier(max_depth=1).fit(X, y) — same feature, same weighted
             child impurity (and the same threshold up to float rounding).
    """
    # TODO: loop over columns, reuse split_candidates, keep the lowest weighted_gini
    Xa = np.asarray(X, dtype=float)
    best = (0, float("nan"), float("inf"))
    for j in range(Xa.shape[1]):
        table = split_candidates(Xa[:, j], y)
        if table.empty:
            continue
        i = int(table["weighted_gini"].to_numpy().argmin())
        g = float(table.loc[i, "weighted_gini"])
        if g < best[2]:
            best = (j, float(table.loc[i, "threshold"]), g)
    return best


# ---------------------------------------------------------------- Task 4
def best_depth(X, y, depths=(1, 2, 3, 4, 5, 6, 8), seed: int = 0) -> tuple[int, pd.DataFrame]:
    """Depth sweep with 5-fold stratified CV (Section 6, "The controls").

    For each depth fit DecisionTreeClassifier(max_depth=d, random_state=seed) and score
    it with cross_val_score(cv=StratifiedKFold(5, shuffle=True, random_state=seed)).
    Also record the TRAINING accuracy of the tree fit on all rows.
    Returns (best_depth, table) where table has columns ["max_depth", "train_acc",
    "cv_acc", "cv_std"] and best_depth is the depth with the highest cv_acc (first on ties).

    Example: on load_wine the sweep peaks at depth 3 (cv ~ 0.94), exactly as in the chapter.
    """
    # TODO: loop over depths, cross_val_score + a full fit for train accuracy, pick argmax of cv mean
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=seed)
        s = cross_val_score(tree, X, y, cv=cv)
        tree.fit(X, y)
        rows.append({"max_depth": d, "train_acc": float(tree.score(X, y)),
                     "cv_acc": float(s.mean()), "cv_std": float(s.std())})
    table = pd.DataFrame(rows)
    best = int(table.loc[table["cv_acc"].idxmax(), "max_depth"])
    return best, table


# ---------------------------------------------------------------- Task 5
def prune_alpha(X_tr, y_tr, seed: int = 0, cv: int = 5):
    """Cost-complexity pruning (Section 6): grow the full tree, then pick ccp_alpha by CV.

    Steps: full = DecisionTreeClassifier(random_state=seed).fit(X_tr, y_tr);
    alphas = full.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas[:-1] (the last alpha
    collapses the tree to its root — skip it); for each alpha compute the mean
    cross_val_score(DecisionTreeClassifier(random_state=seed, ccp_alpha=a), cv=cv); keep
    the alpha with the highest mean (first on ties) and refit with it on all of X_tr.
    Returns (best_alpha, pruned_tree, full_tree).

    Example: on the chapter's make_classification data the full tree has ~62 leaves and
             the pruned tree ~10, with higher held-out accuracy.
    """
    # TODO: fit the full tree, walk the pruning path with cross_val_score, refit at the best alpha
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    full = DecisionTreeClassifier(random_state=seed).fit(X_tr, y_tr)
    alphas = full.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas[:-1]
    best_a, best_score = None, -1.0
    for a in alphas:
        score = cross_val_score(DecisionTreeClassifier(random_state=seed, ccp_alpha=a), X_tr, y_tr, cv=cv).mean()
        if score > best_score:
            best_a, best_score = float(a), float(score)
    pruned = DecisionTreeClassifier(random_state=seed, ccp_alpha=best_a).fit(X_tr, y_tr)
    return best_a, pruned, full


# ---------------------------------------------------------------- Task 6
def importance_report(tree, X_te, y_te, feature_names, n_repeats: int = 20,
                      tol: float = 0.005, seed: int = 0) -> pd.DataFrame:
    """Impurity importance vs permutation importance on HELD-OUT data (Section 7).

    Returns a DataFrame indexed by feature name with columns
      "impurity"     : tree.feature_importances_
      "permutation"  : permutation_importance(tree, X_te, y_te, n_repeats, random_state=seed).importances_mean
      "suspect"      : True where permutation <= tol (the model does not need the feature)
    sorted by "impurity" descending.

    Example: on the collections data `random_id` gets a visible impurity share from a
             fully grown tree but permutation ~0, so it is flagged as suspect; the top
             real driver is not.
    """
    # TODO: build the two Series, compute the suspect flag, sort
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(tree, X_te, y_te, n_repeats=n_repeats, random_state=seed)
    rep = pd.DataFrame({"impurity": tree.feature_importances_,
                        "permutation": perm.importances_mean}, index=list(feature_names))
    rep["suspect"] = rep["permutation"] <= tol
    return rep.sort_values("impurity", ascending=False)


# ---------------------------------------------------------------- Task 7
def segment_table(X, y, feature_names, n_segments: int = 5, min_leaf_frac: float = 0.02,
                  seed: int = 0) -> pd.DataFrame:
    """"Segments, not scores" (Section 9): a max_leaf_nodes tree as a rule table.

    Fit DecisionTreeClassifier(max_leaf_nodes=n_segments,
    min_samples_leaf=ceil(min_leaf_frac * n), random_state=seed). Then walk tree_ to
    produce one row per LEAF with columns:
      "rule"          : the conditions on the path from the root, joined by " and ",
                        each written as "<feature> <= <thr:.3g>" or "<feature> > <thr:.3g>"
      "n"             : training rows in the leaf (tree_.n_node_samples)
      "positive_rate" : share of label 1 in the leaf (from tree_.value)
    sorted by positive_rate descending, index reset.

    Example: on the collections data with n_segments=5 you get 5 rows whose "n" sum to
             1500, every n >= 30, and the top rule mentions late_payments_24m.
    """
    # TODO: fit the tree, then recurse over tree_.children_left/right collecting conditions until a leaf (children == -1)
    from sklearn.tree import DecisionTreeClassifier

    n = len(y)
    tree = DecisionTreeClassifier(max_leaf_nodes=n_segments, min_samples_leaf=math.ceil(min_leaf_frac * n),
                                  random_state=seed).fit(X, y)
    t = tree.tree_
    rows = []

    def walk(node: int, conds: list[str]) -> None:
        if t.children_left[node] == -1:
            counts = t.value[node][0]            # class counts (or fractions, in newer sklearn)
            pos = counts[1] / counts.sum()
            rows.append({"rule": " and ".join(conds) if conds else "all rows",
                         "n": int(t.n_node_samples[node]), "positive_rate": float(pos)})
            return
        name, thr = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{name} <= {thr:.3g}"])
        walk(t.children_right[node], conds + [f"{name} > {thr:.3g}"])

    walk(0, [])
    return pd.DataFrame(rows).sort_values("positive_rate", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from sklearn.datasets import load_wine, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text

    print(f"parent: gini={impurity([5, 5]):.3f} entropy={impurity([5, 5], 'entropy'):.3f}")
    print(split_candidates(CALLS, CHURNED).round(3).to_string(index=False))

    X, y = make_collections_data()
    j, thr, g = best_split(X, y)
    sk = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    print(f"root split from scratch: {FEATURES[j]} <= {thr:.4f} (weighted gini {g:.4f}); "
          f"sklearn: {FEATURES[sk.tree_.feature[0]]} <= {sk.tree_.threshold[0]:.4f}")

    Xw, yw = load_wine(return_X_y=True)
    d, table = best_depth(Xw, yw)
    print(f"wine depth sweep -> best depth {d}\n{table.round(3).to_string(index=False)}")

    Xc, yc = make_classification(n_samples=800, n_features=20, n_informative=6, n_redundant=4, flip_y=0.08, random_state=3)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, random_state=1, stratify=yc)
    alpha, pruned, full = prune_alpha(Xc_tr, yc_tr)
    print(f"pruning: alpha={alpha:.4f}  full leaves={full.get_n_leaves()} test acc={full.score(Xc_te, yc_te):.3f}  "
          f"pruned leaves={pruned.get_n_leaves()} test acc={pruned.score(Xc_te, yc_te):.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    deep = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    print(importance_report(deep, X_te, y_te, FEATURES).round(3))

    print(segment_table(X, y, FEATURES, n_segments=5).to_string(index=False))
    print(export_text(DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(X, y),
                      feature_names=FEATURES, class_names=["paid", "default"]))
