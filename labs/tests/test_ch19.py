import time
from collections import Counter

import numpy as np
import pytest

from labs import ch19_nlp_embeddings as lab


# ---------------------------------------------------------------- Task 1
def test_task1_tokenize_and_vocab():
    assert lab.tokenize("Charged TWICE, need a refund!") == ["charged", "twice", "need", "a", "refund"]
    assert lab.tokenize("Error 500 after v4.2 update") == ["error", "500", "after", "v4", "2", "update"]
    assert lab.build_vocab(["a b b", "b c"]) == {"b": 0, "a": 1, "c": 2}
    v = lab.build_vocab(["x y y z z z", "y w"], min_count=2)
    assert v == {"y": 0, "z": 1}  # ties in count -> alphabetical
    texts, _ = lab.make_ticket_corpus()
    v = lab.build_vocab(texts)
    assert v["the"] == 0 and len(v) == len({t for d in texts for t in lab.tokenize(d)})
    assert list(v.values()) == list(range(len(v)))


# ---------------------------------------------------------------- Task 2
def test_task2_bpe_reproduces_the_chapter_merge_sequence():
    merges, words = lab.bpe_train(lab.make_bpe_corpus(), n_merges=8)
    assert merges == [("l", "o"), ("lo", "w"), ("e", "s"), ("es", "t"), ("est", "</w>"),
                      ("low", "</w>"), ("n", "e"), ("ne", "w")]
    assert isinstance(words, Counter)
    assert dict(words) == {"low</w>": 3, "low e r </w>": 2, "new est</w>": 3, "w i d est</w>": 1, "w i d e </w>": 1}
    vocab = sorted({s for w in words for s in w.split()})
    assert vocab == ["</w>", "d", "e", "est</w>", "i", "low", "low</w>", "new", "r", "w"]
    m3, _ = lab.bpe_train(lab.make_bpe_corpus(), n_merges=3)
    assert m3 == merges[:3]


def test_task2_bpe_merges_only_adjacent_symbols():
    merges, words = lab.bpe_train(["es", "east", "ease"], n_merges=1)
    # ("e","a") occurs twice (east, ease); ("e","s") only once -> the most frequent adjacent pair wins
    assert merges == [("e", "a")]
    assert dict(words) == {"e s </w>": 1, "ea s t </w>": 1, "ea s e </w>": 1}


# ---------------------------------------------------------------- Task 3
def test_task3_bpe_encode_new_words():
    merges, _ = lab.bpe_train(lab.make_bpe_corpus(), n_merges=8)
    assert lab.bpe_encode("lowest", merges) == ["low", "est</w>"]
    assert lab.bpe_encode("newer", merges) == ["new", "e", "r", "</w>"]
    assert lab.bpe_encode("low", merges) == ["low</w>"]
    assert lab.bpe_encode("xyz", merges) == ["x", "y", "z", "</w>"]
    assert lab.bpe_encode("lowlow", merges) == ["low", "low</w>"]


# ---------------------------------------------------------------- Task 4
def test_task4_tfidf_classifier_beats_80_percent_out_of_fold():
    texts, labels = lab.make_ticket_corpus()
    res = lab.tfidf_ticket_classifier(texts, labels, cv=4)
    assert res["cv_accuracy"] > 0.8
    assert set(res["top_terms"]) == {"billing", "bug", "feature"}
    assert all(len(v) == 3 for v in res["top_terms"].values())
    assert "refund" in res["top_terms"]["billing"] or "charged" in res["top_terms"]["billing"]
    assert "error" in res["top_terms"]["bug"]
    assert "add" in res["top_terms"]["feature"]
    new = ["I was billed twice for the same order", "the search page shows an error after login",
           "could you add an option to export as pdf"]
    assert list(res["pipeline"].predict(new)) == ["billing", "bug", "feature"]


def test_task4_cv_accuracy_is_out_of_fold_not_training_accuracy():
    texts, labels = lab.make_ticket_corpus()
    res = lab.tfidf_ticket_classifier(texts, labels, cv=5)
    train_acc = (res["pipeline"].predict(texts) == np.array(labels)).mean()
    assert train_acc > 0.98                    # a TF-IDF + LR model memorises 60 tickets
    assert res["cv_accuracy"] < train_acc - 0.05   # the honest number is lower


# ---------------------------------------------------------------- Task 5
def test_task5_cosine_top_k_uses_direction_not_length():
    rows = np.array([[10.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    idx, sims = lab.cosine_top_k(np.array([1.0, 0.0]), rows, k=3)
    assert idx.tolist() == [0, 1, 2]
    assert sims == pytest.approx([1.0, np.sqrt(0.5), 0.0])
    # a raw dot product would rank the long vector first; cosine must not
    rows2 = np.array([[100.0, 1.0], [0.0, 1.0]])
    idx2, _ = lab.cosine_top_k(np.array([0.0, 1.0]), rows2, k=2)
    assert idx2.tolist() == [1, 0]
    emb = {"king": [0.9, 0.8, 0.1, 0.0], "queen": [0.9, 0.1, 0.8, 0.0], "man": [0.1, 0.9, 0.1, 0.0],
           "woman": [0.1, 0.1, 0.9, 0.0], "pizza": [0.0, 0.1, 0.1, 0.95]}
    names = list(emb)
    M = np.array([emb[w] for w in names])
    target = np.array(emb["king"]) - np.array(emb["man"]) + np.array(emb["woman"])
    idx3, _ = lab.cosine_top_k(target, M, k=2)
    assert names[idx3[0]] == "queen"
    assert len(lab.cosine_top_k(target, M, k=2)[0]) == 2


# ---------------------------------------------------------------- Task 6
def test_task6_skipgram_puts_king_near_queen_not_banana():
    t0 = time.time()
    vecs = lab.train_skipgram(lab.make_word2vec_sentences(), dim=16, epochs=20, seed=0)
    assert time.time() - t0 < 15
    assert set(["king", "queen", "banana", "throne"]) <= set(vecs)
    assert all(np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4) for v in vecs.values())
    cos = lambda a, b: float(vecs[a] @ vecs[b])
    assert cos("king", "queen") > cos("king", "banana") + 0.2
    for fruit in ["banana", "apple", "mango", "cherry"]:
        assert cos("king", "queen") > cos("king", fruit)
    # reproducible with the seed
    vecs2 = lab.train_skipgram(lab.make_word2vec_sentences(), dim=16, epochs=20, seed=0)
    assert np.allclose(vecs["king"], vecs2["king"])


# ---------------------------------------------------------------- Task 7
def test_task7_chunks_cover_all_text_with_the_requested_overlap():
    assert lab.chunk_text("a b c d e f g", size=3, overlap=1) == ["a b c", "c d e", "e f g"]
    words = lab.HELP_CENTRE.split()
    chunks = lab.chunk_text(lab.HELP_CENTRE, size=30, overlap=8)
    assert all(len(c.split()) <= 30 for c in chunks)
    assert " ".join(words) == lab.HELP_CENTRE.strip()
    covered = Counter(w for c in chunks for w in c.split())
    assert all(covered[w] >= 1 for w in words)
    assert chunks[0].split() == words[:30]
    assert chunks[-1].split()[-1] == words[-1]
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-8:] == b.split()[:8]           # exactly `overlap` shared words
    assert len(chunks) == 1 + int(np.ceil((len(words) - 30) / 22))
    with pytest.raises(ValueError):
        lab.chunk_text(lab.HELP_CENTRE, size=10, overlap=10)


def test_task7_retrieval_returns_the_chunk_with_the_answer():
    chunks = lab.chunk_text(lab.HELP_CENTRE, size=30, overlap=8)
    i, score = lab.retrieve_best_chunk("are sms codes supported for two-factor authentication", chunks)
    assert "SMS codes are not supported" in chunks[i]
    assert 0.3 < score <= 1.0
    j, _ = lab.retrieve_best_chunk("how many business days until a refund is returned", chunks)
    assert "5 business days" in chunks[j]
    k, _ = lab.retrieve_best_chunk("which currency are payments settled in", chunks)
    assert "settled in the account currency" in chunks[k]
