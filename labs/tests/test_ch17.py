import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from labs import ch17_cnn as lab


# ---------------------------------------------------------------- Task 1
def test_task1_conv_out_formula():
    assert lab.conv_out(28, 3) == 26
    assert lab.conv_out(28, 3, p=1) == 28
    assert lab.conv_out(28, 2, s=2) == 14
    assert lab.conv_out(224, 7, s=2, p=3) == 112
    assert lab.conv_out(32, 3, d=2) == 28
    # chain: three conv(3, p=1) + pool(2) blocks on 28x28 -> 3x3
    n = 28
    for _ in range(3):
        n = lab.conv_out(lab.conv_out(n, 3, p=1), 2, s=2)
    assert n == 3


def test_task1_conv_out_matches_torch_on_odd_cases():
    x = torch.zeros(1, 1, 31, 31)
    for k, s, p, d in [(3, 2, 0, 1), (5, 3, 2, 1), (3, 1, 0, 3), (2, 2, 0, 1)]:
        out = F.conv2d(x, torch.zeros(1, 1, k, k), stride=s, padding=p, dilation=d)
        assert lab.conv_out(31, k, s, p, d) == out.shape[-1]


# ---------------------------------------------------------------- Task 2
def test_task2_conv_params_by_hand_and_against_torch():
    assert lab.conv_params(1, 16, 3) == 160
    assert lab.conv_params(3, 16, 3) == 448
    assert lab.conv_params(64, 128, 3) == 73856
    assert lab.conv_params(8, 8, 1, bias=False) == 64
    for c_in, c_out, k, bias in [(3, 7, 5, True), (16, 32, 3, False)]:
        layer = nn.Conv2d(c_in, c_out, k, bias=bias)
        assert lab.conv_params(c_in, c_out, k, bias) == sum(p.numel() for p in layer.parameters())


# ---------------------------------------------------------------- Task 3
def test_task3_cross_correlation_matches_the_chapter_example():
    img, ker = lab.make_edge_example()
    out = lab.cross_correlate2d(img, ker)
    assert out.shape == (4, 4)
    assert np.allclose(out, np.array([[0, 3, 3, 0]] * 4))
    assert lab.cross_correlate2d(img, ker, padding=1).shape == (6, 6)


def test_task3_cross_correlation_matches_torch_with_padding_and_stride():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(9, 11))
    ker = rng.normal(size=(3, 4))          # asymmetric: flipping it (true convolution) would fail
    for p, s in [(0, 1), (1, 1), (2, 2), (0, 3)]:
        mine = lab.cross_correlate2d(img, ker, padding=p, stride=s)
        ref = F.conv2d(torch.tensor(img).view(1, 1, 9, 11), torch.tensor(ker).view(1, 1, 3, 4),
                       padding=p, stride=s)[0, 0].numpy()
        assert mine.shape == ref.shape
        assert np.allclose(mine, ref, atol=1e-9)
    # the classic mistake: implementing true convolution (flipped kernel)
    flipped = F.conv2d(torch.tensor(img).view(1, 1, 9, 11), torch.tensor(ker[::-1, ::-1].copy()).view(1, 1, 3, 4))[0, 0].numpy()
    assert not np.allclose(lab.cross_correlate2d(img, ker), flipped)


# ---------------------------------------------------------------- Task 4
def test_task4_max_pool_matches_hand_example_and_torch():
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    assert np.array_equal(lab.max_pool2d(x, 2), np.array([[[[6, 8], [14, 16]]]]))
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 9, 8))
    for k, s in [(2, None), (3, 2), (2, 1), (3, 3)]:
        mine = lab.max_pool2d(x, k, s)
        ref = F.max_pool2d(torch.tensor(x), k, stride=s).numpy()
        assert mine.shape == ref.shape
        assert np.allclose(mine, ref)


# ---------------------------------------------------------------- Task 5
def test_task5_smallcnn_shapes_and_parameter_count():
    torch.manual_seed(0)
    net = lab.SmallCNN(n_classes=10)
    x = torch.randn(4, 1, 28, 28)
    shapes = net.feature_shapes(x)
    assert shapes["block1"] == (4, 16, 14, 14)
    assert shapes["block2"] == (4, 32, 7, 7)
    assert shapes["block3"] == (4, 64, 7, 7)
    assert shapes["gap"] == (4, 64, 1, 1)
    assert shapes["logits"] == (4, 10)
    assert tuple(net(x).shape) == (4, 10)
    assert sum(p.numel() for p in net.parameters()) == 24170
    assert tuple(lab.SmallCNN(n_classes=3)(torch.randn(2, 1, 28, 28)).shape) == (2, 3)


def test_task5_smallcnn_returns_logits_and_uses_batchnorm():
    torch.manual_seed(0)
    net = lab.SmallCNN(n_classes=5).eval()
    out = net(torch.randn(8, 1, 28, 28) * 3)
    # classic mistake: a softmax inside the model -> rows would sum to 1 and be non-negative
    assert not torch.allclose(out.sum(1), torch.ones(8), atol=1e-4) or (out < 0).any()
    assert any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    assert sum(isinstance(m, nn.Conv2d) for m in net.modules()) == 3


# ---------------------------------------------------------------- Task 6
def test_task6_stripes_cnn_learns_in_seconds():
    t0 = time.time()
    res = lab.train_stripes_cnn(n_train=400, n_test=200, epochs=8, seed=0)
    elapsed = time.time() - t0
    assert elapsed < 15
    assert set(res) >= {"test_acc", "train_loss", "test_acc_curve", "n_params"}
    assert len(res["train_loss"]) == len(res["test_acc_curve"]) == 8
    assert res["test_acc"] > 0.9
    assert res["train_loss"][-1] < res["train_loss"][0]
    assert res["n_params"] < 5000       # a tiny net is enough: the signal is local


def test_task6_train_and_test_images_are_different_draws():
    X_a, y_a = lab.make_stripe_images(50, seed=0)
    X_b, y_b = lab.make_stripe_images(50, seed=1)
    assert tuple(X_a.shape) == (50, 1, 16, 16) and y_a.dtype == torch.int64
    assert not torch.allclose(X_a, X_b)
    res = lab.train_stripes_cnn(n_train=200, n_test=100, epochs=6, seed=1)
    assert res["test_acc"] > 0.9
