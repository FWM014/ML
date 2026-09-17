"""
Lab 18 — Attention from scratch (Chapter 18: Sequences, Attention & Transformers)
=================================================================================

THE PROBLEM
-----------
ClauseIQ sells contract review to mid-size law firms. The product team wants to
fine-tune a pretrained Transformer to flag risky clauses, but the last vendor demo went
badly: nobody in the room could explain why the model "saw" a clause 3,000 tokens
away, why the demo model leaked the answer during a next-token test, or what
"8 heads, d_model 512, 4k context" would cost in memory. Before the firm signs a
six-figure fine-tuning contract, the CTO wants the team to build the pieces themselves
on toy sizes: scaled dot-product attention checked against a hand-computed 3-token
example, the causal mask that stops a decoder from peeking, the head split/merge that
makes multi-head attention "free", sinusoidal positions, a real
nn.TransformerEncoderLayer with its parameter count derived by hand, and a tiny
character-level language model that provably learns a repeated sentence. Everything
runs on a laptop CPU in seconds.

TASKS (make the tests in labs/tests/test_ch18.py pass, one at a time)
---------------------------------------------------------------------
1. scaled_dot_product_attention(Q, K, V, mask)  → (output, weights) with the sqrt(d_k) scaling
2. causal_mask(T)                                → lower-triangular (T, T) mask: 1 = may attend
3. split_heads(x, n_heads) / merge_heads(x)      → (B, T, D) <-> (B, h, T, D/h) shape gymnastics
4. sinusoidal_positional_encoding(max_len, d)    → the sin/cos table from §6, no parameters
5. build_encoder(...) / encoder_param_count(...) → token+position embeddings + nn.TransformerEncoder
6. train_char_lm(text, ...)                      → a causal char model whose loss drops and which continues the text

STRETCH (no tests)
------------------
- Remove the causal mask in train_char_lm (Exercise 18.2a). Predict the training loss
  and the generated text *before* you run it. Then remove the positional encoding instead.
- Estimate the KV-cache size per token for a "32 layers, d=4096, fp16" model and how
  many tokens fit in 24 GB (§9). Which of your firm's documents would not fit?

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tiny models: one thread beats thread-sync overhead and keeps timings reproducible.
torch.set_num_threads(1)


def make_three_token_example():
    """The Q, K, V of Figure 18.4 ("the", "cat", "sat"; d_k = 2). Do not modify."""
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return Q, K, V


def make_char_dataset(text: str):
    """Map a string to (data: LongTensor of ids, chars: sorted vocabulary list). Do not modify."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return torch.tensor([stoi[c] for c in text]), chars


# ---------------------------------------------------------------- Task 1
def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """§5: attention = softmax(Q K^T / sqrt(d_k)) V, in NumPy, for 2-D Q (n_q, d_k),
    K (n_k, d_k), V (n_k, d_v).

    - scores = Q @ K.T / sqrt(d_k), where d_k = Q.shape[-1]
    - if mask is given (n_q, n_k), set scores to -inf where mask == 0 BEFORE the softmax
    - weights = softmax over the LAST axis (each row sums to 1); use the max-subtraction
      trick so large scores do not overflow
    - output = weights @ V
    Returns (output (n_q, d_v), weights (n_q, n_k)).

    Example (Figure 18.4): with make_three_token_example(), weights[0] == [0.401, 0.198, 0.401]
    and output[0] == [3.0, 4.0]. Without the sqrt(d_k) you would get [0.422, 0.155, 0.422].
    """
    Q, K, V = np.asarray(Q, float), np.asarray(K, float), np.asarray(V, float)
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(np.asarray(mask) == 0, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / e.sum(axis=-1, keepdims=True)
    return weights @ V, weights


# ---------------------------------------------------------------- Task 2
def causal_mask(T: int) -> np.ndarray:
    """§8: an int array of shape (T, T) with 1 where query i may attend to key j (j <= i)
    and 0 elsewhere — i.e. np.tril(np.ones((T, T))).

    Example: causal_mask(3) == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    """
    return np.tril(np.ones((T, T), dtype=int))


# ---------------------------------------------------------------- Task 3
def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """§6: reshape (B, T, D) -> (B, n_heads, T, D // n_heads) so every head gets its own
    contiguous slice of the model width. D must be divisible by n_heads (raise ValueError).

    Example: x of shape (2, 5, 16) with 4 heads -> (2, 4, 5, 4), and
    split_heads(x, 4)[:, 0] == x[..., :4].
    """
    B, T, D = x.shape
    if D % n_heads:
        raise ValueError(f"D={D} not divisible by n_heads={n_heads}")
    return x.reshape(B, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """Inverse of split_heads: (B, h, T, d_head) -> (B, T, h * d_head), concatenating the
    heads back in order so merge_heads(split_heads(x, h)) == x exactly.
    """
    B, h, T, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, h * d)


# ---------------------------------------------------------------- Task 4
def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """§6: PE[pos, 2i] = sin(pos / 10000^(2i/d_model)), PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model)).

    Returns a float array (max_len, d_model). d_model must be even (raise ValueError).

    Example: PE[0] == [0, 1, 0, 1, ...]; PE[1, 0] == sin(1); PE[1, 1] == cos(1);
    every (sin, cos) pair has unit norm.
    """
    if d_model % 2:
        raise ValueError("d_model must be even")
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angle = pos / np.power(10000.0, i / d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


# ---------------------------------------------------------------- Task 5
class TokenPosEncoder(nn.Module):
    """Token embedding + learned position embedding + nn.TransformerEncoder. Do not modify:
    build_encoder() must construct and return this class."""

    def __init__(self, vocab: int, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=0.0, batch_first=True)
        self.stack = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:      # ids: (B, T) -> (B, T, d_model)
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T))
        return self.stack(x)


def build_encoder(vocab: int = 1000, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 256,
                  num_layers: int = 3, max_len: int = 12, seed: int = 0) -> nn.Module:
    """§7: torch.manual_seed(seed), then return a TokenPosEncoder(...) built with these sizes.

    Its forward maps LongTensor ids (B, T) with T <= max_len to (B, T, d_model): the shape
    is preserved through every block, which is what lets blocks stack.
    """
    torch.manual_seed(seed)
    return TokenPosEncoder(vocab, d_model, nhead, dim_feedforward, num_layers, max_len)


def encoder_param_count(d_model: int, dim_feedforward: int, num_layers: int) -> int:
    """Parameters in nn.TransformerEncoder(num_layers) computed BY HAND (no torch):

    per layer = attention (in-projection W of 3*d*d + 3*d bias, out-projection d*d + d)
              + feed-forward (d*ff + ff, ff*d + d)
              + two LayerNorms (2*d each).
    Example: d_model=64, dim_feedforward=256, num_layers=3 -> 149,952 (matches the chapter).
    """
    d, ff = d_model, dim_feedforward
    attn = 3 * d * d + 3 * d + d * d + d
    ffn = d * ff + ff + ff * d + d
    norms = 2 * (2 * d)
    return num_layers * (attn + ffn + norms)


# ---------------------------------------------------------------- Task 6
class CharLM(nn.Module):
    """Decoder-style char model: token + position embeddings, ONE TransformerEncoderLayer
    used with a causal mask, then a linear head to next-char logits. (Solution helper.)"""

    def __init__(self, vocab: int, context: int, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.block = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                                                dropout=0.0, batch_first=True)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:      # (B, t) -> (B, t, vocab)
        B, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t))
        mask = torch.tensor(causal_mask(t)) == 0                 # True = blocked
        x = self.block(x, src_mask=mask)
        return self.head(x)


def train_char_lm(text: str, steps: int = 150, context: int = 16, d_model: int = 32,
                  batch_size: int = 32, lr: float = 3e-3, prompt: str | None = None,
                  n_generate: int = 30, seed: int = 0) -> dict:
    """§10: train a tiny CAUSAL next-character model on `text` and generate a continuation.

    Recipe: torch.manual_seed(seed); (data, chars) = make_char_dataset(text); model =
    token embedding + position embedding + one TransformerEncoderLayer(batch_first=True,
    dropout=0) applied with a causal mask (src_mask: True = blocked) + Linear head to
    len(chars) logits. AdamW(lr). Each step: sample batch_size random windows
    data[j:j+context] with targets data[j+1:j+context+1]; cross_entropy over all positions.
    Then generate greedily from `prompt` (default text[:context]) for n_generate chars,
    feeding at most the last `context` chars each time.

    Returns {"initial_loss": float, "final_loss": float, "losses": [per-step floats],
             "generated": prompt + continuation (str), "vocab_size": int}.
    A working model's final loss is far below the initial one (~ln(vocab)), and on a
    repeated sentence the greedy continuation reproduces the text.
    """
    torch.manual_seed(seed)
    data, chars = make_char_dataset(text)
    V = len(chars)
    model = CharLM(V, context, d_model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        i = torch.randint(0, len(data) - context - 1, (batch_size,))
        xb = torch.stack([data[j:j + context] for j in i])
        yb = torch.stack([data[j + 1:j + context + 1] for j in i])
        loss = F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    prompt = text[:context] if prompt is None else prompt
    stoi = {c: i for i, c in enumerate(chars)}
    idx = torch.tensor([[stoi[c] for c in prompt]])
    model.eval()
    with torch.no_grad():
        for _ in range(n_generate):
            logits = model(idx[:, -context:])[:, -1]
            idx = torch.cat([idx, logits.argmax(-1, keepdim=True)], dim=1)
    generated = "".join(chars[i] for i in idx[0].tolist())
    return {"initial_loss": losses[0], "final_loss": losses[-1], "losses": losses,
            "generated": generated, "vocab_size": V}


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    Q, K, V = make_three_token_example()
    out, w = scaled_dot_product_attention(Q, K, V)
    print("attention weights (rows sum to 1):\n", w, "\noutput:\n", out)
    out_c, w_c = scaled_dot_product_attention(Q, K, V, mask=causal_mask(3))
    print("causal weights (upper triangle is 0):\n", w_c)
    x = np.random.default_rng(0).normal(size=(2, 5, 16))
    print("split_heads:", split_heads(x, 4).shape, "-> merge back equal:", np.array_equal(merge_heads(split_heads(x, 4)), x))
    pe = sinusoidal_positional_encoding(50, 8)
    print("positional encoding (first 3 positions):\n", pe[:3])
    enc = build_encoder()
    ids = torch.randint(0, 1000, (4, 12))
    print("encoder:", tuple(ids.shape), "->", tuple(enc(ids).shape),
          "| stack params:", sum(p.numel() for p in enc.stack.parameters()), "by hand:", encoder_param_count(64, 256, 3))
    t0 = time.time()
    res = train_char_lm("the quick brown fox jumps over the lazy dog. " * 40, steps=150)
    print(f"char LM: loss {res['initial_loss']:.2f} -> {res['final_loss']:.2f} in {time.time() - t0:.1f}s")
    print("generated:", repr(res["generated"]))
