import math
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from labs import ch18_attention as lab


# ---------------------------------------------------------------- Task 1
def test_task1_attention_matches_the_hand_computed_three_token_example():
    Q, K, V = lab.make_three_token_example()
    out, w = lab.scaled_dot_product_attention(Q, K, V)
    assert w.shape == (3, 3) and out.shape == (3, 2)
    expected_w = np.array([[0.401, 0.198, 0.401], [0.198, 0.401, 0.401], [0.248, 0.248, 0.503]])
    assert np.allclose(w, expected_w, atol=2e-3)
    assert np.allclose(out, [[3.0, 4.0], [3.407, 4.407], [3.510, 4.510]], atol=2e-3)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_task1_the_sqrt_dk_scaling_is_applied():
    # classic mistake: forgetting / sqrt(d_k). Unscaled weights on row 0 would be [0.422, 0.155, 0.422].
    Q, K, V = lab.make_three_token_example()
    _, w = lab.scaled_dot_product_attention(Q, K, V)
    assert abs(w[0, 1] - 0.198) < 5e-3 and abs(w[0, 1] - 0.155) > 3e-2
    # exercise 18.1: doubling the "cat" query sharpens its row
    Q2 = Q.copy()
    Q2[1] = [0.0, 2.0]
    _, w2 = lab.scaled_dot_product_attention(Q2, K, V)
    assert np.allclose(w2[1], [0.11, 0.45, 0.45], atol=1e-2)


def test_task1_mask_and_numerical_stability():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 3))
    mask = np.ones((4, 6), dtype=int)
    mask[:, 4:] = 0                                  # nobody may look at the last two keys
    out, w = lab.scaled_dot_product_attention(Q, K, V, mask=mask)
    assert out.shape == (4, 3)
    assert np.all(w[:, 4:] == 0) and np.allclose(w.sum(1), 1.0)
    # huge scores must not overflow into NaN
    _, w_big = lab.scaled_dot_product_attention(Q * 1e3, K * 1e3, V)
    assert np.isfinite(w_big).all() and np.allclose(w_big.sum(1), 1.0)


# ---------------------------------------------------------------- Task 2
def test_task2_causal_mask_blocks_the_future():
    m = lab.causal_mask(5)
    assert m.shape == (5, 5)
    assert np.array_equal(m, np.tril(np.ones((5, 5))))
    assert np.array_equal(lab.causal_mask(3), [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    rng = np.random.default_rng(1)
    Q, K, V = rng.normal(size=(5, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 2))
    _, w = lab.scaled_dot_product_attention(Q, K, V, mask=lab.causal_mask(5))
    assert np.all(np.triu(w, k=1) == 0)
    assert w[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------- Task 3
def test_task3_split_and_merge_heads_roundtrip():
    x = np.random.default_rng(0).normal(size=(2, 5, 16))
    h = lab.split_heads(x, 4)
    assert h.shape == (2, 4, 5, 4)
    assert np.array_equal(h[:, 0], x[..., :4])
    assert np.array_equal(h[:, 3], x[..., 12:])
    assert np.array_equal(lab.merge_heads(h), x)
    assert lab.merge_heads(lab.split_heads(x, 8)).shape == (2, 5, 16)
    with pytest.raises(ValueError):
        lab.split_heads(x, 3)


# ---------------------------------------------------------------- Task 4
def test_task4_sinusoidal_positional_encoding_matches_the_formula():
    pe = lab.sinusoidal_positional_encoding(50, 8)
    assert pe.shape == (50, 8)
    assert np.allclose(pe[0], [0, 1, 0, 1, 0, 1, 0, 1])
    assert pe[1, 0] == pytest.approx(math.sin(1.0))
    assert pe[1, 1] == pytest.approx(math.cos(1.0))
    for pos, i in [(7, 1), (23, 3), (49, 0)]:
        wavelength = 10000 ** (2 * i / 8)
        assert pe[pos, 2 * i] == pytest.approx(math.sin(pos / wavelength))
        assert pe[pos, 2 * i + 1] == pytest.approx(math.cos(pos / wavelength))
    assert np.allclose(pe[:, 0::2] ** 2 + pe[:, 1::2] ** 2, 1.0)
    assert np.abs(pe).max() <= 1.0
    # nearby positions get similar vectors, distant ones less so
    d = lambda a, b: np.linalg.norm(pe[a] - pe[b])
    assert d(10, 11) < d(10, 30)
    with pytest.raises(ValueError):
        lab.sinusoidal_positional_encoding(10, 7)


# ---------------------------------------------------------------- Task 5
def test_task5_encoder_preserves_shape_and_param_count_is_derived_by_hand():
    enc = lab.build_encoder(vocab=1000, d_model=64, nhead=4, dim_feedforward=256, num_layers=3, max_len=12, seed=0)
    assert isinstance(enc, nn.Module)
    ids = torch.randint(0, 1000, (4, 12))
    out = enc(ids)
    assert tuple(out.shape) == (4, 12, 64)
    assert tuple(enc(ids[:, :7]).shape) == (4, 7, 64)
    stack = [m for m in enc.modules() if isinstance(m, nn.TransformerEncoder)][0]
    assert sum(p.numel() for p in stack.parameters()) == 149952
    assert lab.encoder_param_count(64, 256, 3) == 149952
    for d, ff, L in [(32, 64, 1), (16, 128, 2)]:
        e = lab.build_encoder(vocab=50, d_model=d, nhead=4, dim_feedforward=ff, num_layers=L, max_len=8)
        st = [m for m in e.modules() if isinstance(m, nn.TransformerEncoder)][0]
        assert lab.encoder_param_count(d, ff, L) == sum(p.numel() for p in st.parameters())


def test_task5_build_encoder_is_seeded():
    ids = torch.randint(0, 100, (2, 6))
    a = lab.build_encoder(vocab=100, d_model=16, nhead=2, dim_feedforward=32, num_layers=1, max_len=6, seed=3)(ids)
    b = lab.build_encoder(vocab=100, d_model=16, nhead=2, dim_feedforward=32, num_layers=1, max_len=6, seed=3)(ids)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------- Task 6
def test_task6_char_lm_loss_drops_and_continues_the_text():
    text = "the quick brown fox jumps over the lazy dog. " * 40
    t0 = time.time()
    res = lab.train_char_lm(text, steps=150, context=16, seed=0)
    assert time.time() - t0 < 20
    assert res["vocab_size"] == len(set(text))
    assert res["initial_loss"] > 2.5                       # ~ ln(vocab) at init
    assert res["final_loss"] < 0.4
    assert res["final_loss"] < 0.2 * res["initial_loss"]
    assert len(res["losses"]) == 150
    gen = res["generated"]
    assert gen.startswith(text[:16])
    # greedy generation must reproduce the repeated sentence: the causal mask + positions work
    assert gen[16:16 + 25] == text[16:16 + 25]


def test_task6_char_lm_on_a_short_cycle():
    text = "abcdefgh " * 30
    res = lab.train_char_lm(text, steps=100, context=8, n_generate=20, seed=1)
    assert res["final_loss"] < 0.2
    assert res["generated"] == text[:8 + 20]
