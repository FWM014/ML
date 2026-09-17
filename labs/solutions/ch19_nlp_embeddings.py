"""
Lab 19 — Text to numbers to answers (Chapter 19: NLP, Embeddings & Large Language Models)
=========================================================================================

THE PROBLEM
-----------
Ledgerly, a B2B invoicing start-up, receives ~900 support tickets a week that three
people triage by hand into billing / bug / feature. The CEO wants "an LLM for support"
by next quarter; the CTO wants proof that the cheap, explainable pieces work first, and
a RAG prototype over the help-centre pages so the LLM — when it comes — answers from the
docs instead of guessing. You have no GPU, no network, and no Hugging Face: everything
must run offline in seconds. You will write a tokenizer and vocabulary, a byte-pair
encoder that reproduces the chapter's merge sequence, a TF-IDF + logistic-regression
classifier that beats 80 % cross-validated accuracy on an inline 60-ticket corpus, cosine
top-k search over hand-made embeddings, a tiny skip-gram word2vec in PyTorch that
learns "king" is nearer "queen" than "banana", and the two functions every RAG system
needs: an overlapping chunker that loses no text, and a TF-IDF retriever that returns
the best chunk for a question.

TASKS (make the tests in labs/tests/test_ch19.py pass, one at a time)
---------------------------------------------------------------------
1. tokenize(text) / build_vocab(docs, min_count)   → lowercase word tokens; token -> id by frequency
2. bpe_train(corpus, n_merges)                     → the merge list of §2, exactly as the chapter prints it
3. bpe_encode(word, merges)                        → segment a NEW word with a learned merge list
4. tfidf_ticket_classifier(texts, labels)          → TF-IDF + LogisticRegression, CV accuracy > 0.8
5. cosine_top_k(query, matrix, k)                  → indices of the k most similar rows (cosine, not dot)
6. train_skipgram(sentences, dim, epochs, seed)    → word -> unit vector; royalty clusters away from fruit
7. chunk_text(text, size, overlap) / retrieve_best_chunk(query, chunks) → RAG offline + online halves

STRETCH (no tests)
------------------
- Swap TfidfVectorizer for the skip-gram vectors (average per ticket) in task 4. Which
  wins on 60 tickets, and why would the answer change at 60,000?
- Write the system prompt + JSON output schema (§8) for the ticket triage LLM and list
  the three golden-set checks you would run before it replaces task 4.

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Tiny models: one thread beats thread-sync overhead and keeps timings reproducible.
torch.set_num_threads(1)


def make_bpe_corpus() -> list[str]:
    """The ten-word corpus of §2. Do not modify."""
    return ["low", "low", "low", "lower", "lower", "newest", "newest", "newest", "widest", "wide"]


def make_ticket_corpus() -> tuple[list[str], list[str]]:
    """60 labelled support tickets, 20 per class (billing / bug / feature). Do not modify."""
    billing = [
        "the invoice was paid late and the fee is wrong", "billing charged me twice this month",
        "refund still not received after two weeks", "why is my bill higher than the quoted price",
        "double charge on my card, need a refund", "the subscription price changed without notice",
        "invoice total does not match the contract", "was charged for a plan I cancelled last month",
        "please send a receipt for the annual payment", "vat is missing from the invoice pdf",
        "my credit card was declined but you still billed me", "how do I update the billing address on invoices",
        "the discount code was not applied to my payment", "charged in usd instead of eur on the last invoice",
        "I need a refund for the unused months of my subscription", "the pro-rated charge on the upgrade looks wrong",
        "the payment failed but the invoice shows as paid twice", "refund the duplicate charge from last week please",
        "why does the receipt show a different amount than the invoice", "my card was charged after I downgraded the subscription",
    ]
    bug = [
        "the app crashes when I open the settings page", "login button does nothing on android",
        "error 500 after the latest update", "the dashboard shows blank charts since yesterday",
        "app freezes on the checkout screen", "crash on startup after upgrading to version 4",
        "the export button throws an error every time", "sync fails with a timeout error on wifi",
        "search returns no results even for exact matches", "the pdf preview is blank in safari",
        "notifications stopped working after the update", "the page reloads endlessly on the reports tab",
        "uploaded logo appears stretched in the header", "date picker shows the wrong month",
        "the api returns a 502 error for every request", "the invoice list does not load, spinner forever",
        "the save button crashes the app on ipad", "error message appears when I try to log in",
        "the report page is broken and shows a blank screen", "the app freezes every time I click export",
    ]
    feature = [
        "can you add dark mode to the mobile app", "it would be great to export reports to excel",
        "please support two-factor authentication", "feature request: recurring invoices",
        "would love an api for bulk uploads", "add the option to schedule emails",
        "it would help to have a weekly summary option", "please add support for multiple currencies",
        "could you add a customer portal for self-service", "request: custom fields on invoice templates",
        "it would be nice to tag invoices by project", "please add keyboard shortcuts to the editor",
        "can we get a slack integration for new payments", "would like to set reminders before due dates",
        "add an option to duplicate an existing invoice", "feature idea: a read-only role for accountants",
        "it would be great if invoices could be sent automatically", "please add a way to archive old customers",
        "would love a mobile widget for unpaid invoices", "can you add an integration with quickbooks",
    ]
    return billing + bug + feature, ["billing"] * 20 + ["bug"] * 20 + ["feature"] * 20


def make_word2vec_sentences(seed: int = 0) -> list[str]:
    """A toy corpus where royalty words and fruit words appear in separate contexts. Do not modify."""
    royals = ["king", "queen", "prince", "princess"]
    fruits = ["banana", "apple", "mango", "cherry"]
    royal_tpl = ["the {} ruled the kingdom", "the {} sat on the throne", "a {} wore the crown", "the {} lives in the castle"]
    fruit_tpl = ["i ate a ripe {} today", "a sweet {} for breakfast", "she peeled the {} slowly", "{} tastes good with yogurt"]
    sentences = [t.format(w) for _ in range(15) for t in royal_tpl for w in royals] + \
                [t.format(w) for _ in range(15) for t in fruit_tpl for w in fruits]
    rng = np.random.default_rng(seed)
    rng.shuffle(sentences)
    return sentences


HELP_CENTRE = (
    "Refund policy. Monthly plans can be cancelled at any time and are not refunded for the "
    "current month. Annual plans can be refunded in full within 30 days of the first charge; "
    "after 30 days refunds are pro-rated for the unused months. Refunds are returned to the "
    "original payment method within 5 business days. "
    "Two-factor authentication. Enable two-factor authentication from Settings, Security. We "
    "support authenticator apps and hardware keys; SMS codes are not supported. Recovery codes "
    "are shown once, so store them in a password manager. "
    "Exporting data. Reports can be exported as CSV or PDF from the Reports tab. Exports of more "
    "than 10,000 rows are emailed as a download link within an hour. Excel export is on the roadmap. "
    "Currencies. Invoices can be issued in 40 currencies; the exchange rate is fixed at the time "
    "the invoice is sent. Payments are settled in the account currency chosen at signup."
)


# ---------------------------------------------------------------- Task 1
def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer: re.findall(r"[a-z0-9]+", text.lower()).

    Example: tokenize("Charged TWICE, need a refund!") -> ["charged", "twice", "need", "a", "refund"]
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(docs: list[str], min_count: int = 1) -> dict[str, int]:
    """Token -> integer id over all docs (tokenize each), keeping tokens with count >= min_count,
    ids assigned in order of decreasing count, ties broken alphabetically (ids start at 0).

    Example: build_vocab(["a b b", "b c"]) -> {"b": 0, "a": 1, "c": 2}
    """
    counts = Counter(t for d in docs for t in tokenize(d))
    kept = sorted((t for t, c in counts.items() if c >= min_count), key=lambda t: (-counts[t], t))
    return {t: i for i, t in enumerate(kept)}


# ---------------------------------------------------------------- Task 2
def bpe_train(corpus: list[str], n_merges: int = 8) -> tuple[list[tuple[str, str]], Counter]:
    """§2, exactly as the chapter's code block:

    words = Counter(" ".join(list(w)) + " </w>" for w in corpus)   # "l o w </w>": 3, ...
    repeat n_merges times: count adjacent symbol pairs (weighted by word count), pick
    best = max(pairs, key=pairs.get) (ties -> the FIRST pair encountered in iteration order),
    replace "a b" by "ab" in every word, append best to merges.

    Returns (merges, words). On make_bpe_corpus() the first merges are
    ('l','o'), ('lo','w'), ('e','s'), ('es','t'), ('est','</w>'), ('low','</w>'), ('n','e'), ('ne','w').
    """
    words = Counter(" ".join(list(w)) + " </w>" for w in corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        pairs: Counter = Counter()
        for w, n in words.items():
            syms = w.split()
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        a, b = best
        words = Counter({w.replace(f"{a} {b}", a + b): n for w, n in words.items()})
        merges.append(best)
    return merges, words


# ---------------------------------------------------------------- Task 3
def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Segment a new word with a learned merge list: start from list(word) + ["</w>"], then
    apply every merge IN ORDER, replacing each adjacent (a, b) occurrence by a + b
    (scan left to right; repeat within the word while the pair still occurs).

    Example: with the 8 merges of make_bpe_corpus(), bpe_encode("lowest") -> ["low", "est</w>"]
    and bpe_encode("newer") -> ["new", "e", "r", "</w>"].
    """
    syms = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(syms) - 1:
            if syms[i] == a and syms[i + 1] == b:
                syms[i:i + 2] = [a + b]
            else:
                i += 1
    return syms


# ---------------------------------------------------------------- Task 4
def tfidf_ticket_classifier(texts: list[str], labels: list[str], cv: int = 4) -> dict:
    """§3: Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
    stop_words="english")), ("lr", LogisticRegression(C=10, max_iter=1000))]).

    cv_accuracy = cross_val_score(pipe, texts, labels, cv=cv).mean() (honest, out-of-fold);
    then fit on everything and read the top-3 positive terms per class from lr.coef_.
    Returns {"cv_accuracy": float, "pipeline": fitted Pipeline, "top_terms": {label (str): [3 str terms]}}.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("lr", LogisticRegression(C=10, max_iter=1000)),
    ])
    cv_acc = float(cross_val_score(pipe, texts, labels, cv=cv).mean())
    pipe.fit(texts, labels)
    vocab = pipe["tfidf"].get_feature_names_out()
    top = {str(cls): [str(vocab[i]) for i in row.argsort()[-3:][::-1]]
           for cls, row in zip(pipe["lr"].classes_, pipe["lr"].coef_)}
    return {"cv_accuracy": cv_acc, "pipeline": pipe, "top_terms": top}


# ---------------------------------------------------------------- Task 5
def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """§4: cosine similarity between query (d,) and every row of matrix (n, d); return
    (indices of the k most similar rows, best first; their similarities).

    Cosine compares DIRECTIONS: normalise both sides by their L2 norm (a raw dot product
    would favour long vectors). Example: query [1, 0], rows [[10, 0], [1, 1], [0, 1]] ->
    indices [0, 1, 2], similarities [1.0, 0.707, 0.0].
    """
    q = np.asarray(query, dtype=float)
    M = np.asarray(matrix, dtype=float)
    sims = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-12)
    order = np.argsort(-sims)[:k]
    return order, sims[order]


# ---------------------------------------------------------------- Task 6
def train_skipgram(sentences: list[str], dim: int = 16, window: int = 2, epochs: int = 20,
                   lr: float = 0.02, batch_size: int = 256, seed: int = 0) -> dict[str, np.ndarray]:
    """§4: word2vec skip-gram with a full softmax, as in the chapter's code block.

    - vocab = sorted set of tokens (split on spaces); (center, context) pairs for every
      position and every other token within `window`
    - model: nn.Embedding(V, dim) for centers ("inp") and nn.Embedding(V, dim) for contexts;
      logits = inp(center) @ out.weight.T; F.cross_entropy(logits, context)
    - torch.manual_seed(seed); Adam(lr); `epochs` passes over shuffled pairs in mini-batches
    Returns {word: unit-length numpy vector of the CENTER embedding}. On make_word2vec_sentences()
    the nearest neighbours of "king" are the other royals, and cos(king, queen) > cos(king, banana).
    """
    torch.manual_seed(seed)
    tokens = [s.split() for s in sentences]
    vocab = sorted({w for s in tokens for w in s})
    stoi = {w: i for i, w in enumerate(vocab)}
    pairs = []
    for s in tokens:
        ids = [stoi[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((c, ids[j]))
    pairs = torch.tensor(pairs)
    inp, out = nn.Embedding(len(vocab), dim), nn.Embedding(len(vocab), dim)
    opt = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=lr)
    for _ in range(epochs):
        perm = pairs[torch.randperm(len(pairs))]
        for b in range(0, len(perm), batch_size):
            c, ctx = perm[b:b + batch_size, 0], perm[b:b + batch_size, 1]
            loss = F.cross_entropy(inp(c) @ out.weight.T, ctx)
            opt.zero_grad()
            loss.backward()
            opt.step()
    E = F.normalize(inp.weight.detach(), dim=1).numpy()
    return {w: E[i] for w, i in stoi.items()}


# ---------------------------------------------------------------- Task 7
def chunk_text(text: str, size: int = 40, overlap: int = 10) -> list[str]:
    """§8 (RAG, offline half): split on whitespace into windows of `size` words, starting every
    size - overlap words, so consecutive chunks share `overlap` words and no word is lost.
    The last chunk may be shorter. Raise ValueError if overlap >= size.

    Example: chunk_text("a b c d e f g", size=3, overlap=1) -> ["a b c", "c d e", "e f g"]
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return chunks


def retrieve_best_chunk(query: str, chunks: list[str]) -> tuple[int, float]:
    """§8 (RAG, online half): TfidfVectorizer fitted on the chunks; transform the query with the
    SAME vectorizer; cosine similarity (TF-IDF rows are already L2-normalised, so it is a dot
    product); return (index of the best chunk, its similarity).

    Example: for HELP_CENTRE chunked with size=30, overlap=8, the query
    "are sms codes supported for two-factor authentication" returns the chunk containing
    "SMS codes are not supported" with similarity ≈ 0.62.
    """
    vec = TfidfVectorizer().fit(chunks)
    C = vec.transform(chunks)
    q = vec.transform([query])
    sims = (C @ q.T).toarray().ravel()
    best = int(np.argmax(sims))
    return best, float(sims[best])


if __name__ == "__main__":
    print("tokens:", tokenize("Charged TWICE, need a refund!"), "| vocab:", build_vocab(["a b b", "b c"]))
    merges, words = bpe_train(make_bpe_corpus(), 8)
    print("BPE merges:", merges)
    print("encode 'lowest':", bpe_encode("lowest", merges), "| 'newer':", bpe_encode("newer", merges))
    texts, labels = make_ticket_corpus()
    res = tfidf_ticket_classifier(texts, labels)
    print(f"TF-IDF + LR: CV accuracy {res['cv_accuracy']:.3f}; top terms {res['top_terms']}")
    print("new tickets ->", list(res["pipeline"].predict(["I was billed twice for the same order",
                                                          "the search page shows an error after login",
                                                          "could you add an option to export as pdf"])))
    emb = np.array([[0.9, 0.8, 0.1, 0.0], [0.9, 0.1, 0.8, 0.0], [0.1, 0.9, 0.1, 0.0], [0.0, 0.1, 0.1, 0.95]])
    print("cosine top-2 for 'king':", cosine_top_k(emb[0], emb[1:], k=2))
    vecs = train_skipgram(make_word2vec_sentences())
    cos = lambda a, b: float(vecs[a] @ vecs[b])
    print(f"skip-gram: cos(king, queen) = {cos('king', 'queen'):.2f}, cos(king, banana) = {cos('king', 'banana'):.2f}")
    chunks = chunk_text(HELP_CENTRE, size=30, overlap=8)
    for q in ["are sms codes supported for two-factor authentication", "how many business days until a refund is returned"]:
        i, s = retrieve_best_chunk(q, chunks)
        print(f"RAG: {len(chunks)} chunks; best for '{q}' = #{i} (cos {s:.2f}): ...{chunks[i][-60:]}")
