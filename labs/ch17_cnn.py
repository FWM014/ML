"""
Lab 17 — Convolutions you can verify (Chapter 17: Convolutional Networks & Vision)
=================================================================================

THE PROBLEM
-----------
Halden Packaging prints 2 million cartons a week. A line camera photographs every
carton; a misaligned print head produces horizontal streaks instead of the vertical
grain of a correct print. The vendor's "AI inspection" quote is €180k, so the plant
manager asks you to prove the idea on the cheap first: 16×16 grayscale crops, a
convolutional network trained on a laptop, and — because the safety auditor will ask —
every operation checked against a from-scratch implementation. You will write 2-D
cross-correlation and max-pooling in NumPy and match torch exactly, derive output
sizes and parameter counts by hand, build a small CNN with the conv → BN → ReLU → pool
pattern and verify its shapes on a 1×28×28 batch, and train it on synthetic
vertical-vs-horizontal stripe images (the same signal as the streak defect) to > 90 %
held-out accuracy in a few seconds.

TASKS (make the tests in labs/tests/test_ch17.py pass, one at a time)
---------------------------------------------------------------------
1. conv_out(n, k, s, p, d)                  → the output-size formula of §3
2. conv_params(c_in, c_out, k, bias)        → parameter count of one conv layer
3. cross_correlate2d(img, kernel, p, s)     → NumPy 2-D cross-correlation == F.conv2d
4. max_pool2d(x, k, s)                      → NumPy max pooling on (B, C, H, W) == F.max_pool2d
5. SmallCNN(n_classes)                      → conv-BN-ReLU-pool blocks + GAP + linear head, logits out
6. train_stripes_cnn(...)                   → > 0.9 test accuracy on the stripe task in seconds

STRETCH (no tests)
------------------
- Add a fourth block to SmallCNN and predict the parameter count before running it
  (Exercise 17.2). Then feed a 3×64×64 batch: which layer had to change?
- Rotate the test images by 90° in train_stripes_cnn. What happens to accuracy, and
  which augmentation would you forbid on this task (§7)?

Run this file directly (F5) to print your results as you go.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tiny models: one thread beats thread-sync overhead and keeps timings reproducible.
torch.set_num_threads(1)


def make_edge_example():
    """The 6x6 vertical-edge image and 3x3 edge kernel of §2. Do not modify."""
    img = np.array([[0, 0, 0, 1, 1, 1]] * 6, dtype=np.float32)
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    return img, kernel


def make_stripe_images(n: int, size: int = 16, seed: int = 0):
    """Synthetic carton crops: 0 = vertical stripes (good print), 1 = horizontal (streak).

    Returns X (n, 1, size, size) float32 tensor with noise, y (n,) int64 tensor. Do not modify.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    y = rng.integers(0, 2, n)
    for i in range(n):
        period = rng.integers(3, 6)
        phase = rng.integers(0, period)
        idx = (np.arange(size) + phase) % period < period // 2
        stripes = np.tile(idx, (size, 1)) if y[i] == 0 else np.tile(idx[:, None], (1, size))
        X[i, 0] = stripes + 0.4 * rng.standard_normal((size, size))
    return torch.tensor(X), torch.tensor(y, dtype=torch.int64)


# ---------------------------------------------------------------- Task 1
def conv_out(n: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:
    """§3: spatial size after a conv/pool layer: floor((n + 2p - d(k-1) - 1) / s) + 1.

    n: input size, k: kernel, s: stride, p: padding (each side), d: dilation.
    Examples: conv_out(28, 3) == 26; conv_out(28, 3, p=1) == 28; conv_out(28, 2, s=2) == 14;
    conv_out(224, 7, s=2, p=3) == 112; conv_out(32, 3, d=2) == 28.
    """
    # TODO: return (n + 2p - d(k-1) - 1) // s + 1
    raise NotImplementedError("Task: conv_out")


# ---------------------------------------------------------------- Task 2
def conv_params(c_in: int, c_out: int, k: int, bias: bool = True) -> int:
    """§3: a conv layer has c_out filters of size c_in x k x k, plus c_out biases:
    c_out * c_in * k * k (+ c_out if bias). Independent of the image size.

    Examples: conv_params(1, 16, 3) == 160; conv_params(3, 16, 3) == 448;
    conv_params(64, 128, 3) == 73856; conv_params(8, 8, 1, bias=False) == 64.
    """
    # TODO: c_out * c_in * k * k, plus c_out biases when bias is True
    raise NotImplementedError("Task: conv_params")


# ---------------------------------------------------------------- Task 3
def cross_correlate2d(img: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    """§2: slide the kernel over the (zero-padded) image; each output cell is the elementwise
    product of the kernel and the patch beneath it, summed. This is what F.conv2d computes
    (cross-correlation: the kernel is NOT flipped).

    img: (H, W), kernel: (kH, kW). Output shape: (conv_out(H, kH, stride, padding),
    conv_out(W, kW, stride, padding)). Use np.pad for the padding.

    Example: make_edge_example() -> a 4x4 output whose every row is [0, 3, 3, 0].
    """
    # TODO: np.pad if padding > 0; loop over output cells; each cell = (patch * kernel).sum() with the patch starting at (i*stride, j*stride)
    raise NotImplementedError("Task: cross_correlate2d")


# ---------------------------------------------------------------- Task 4
def max_pool2d(x: np.ndarray, k: int = 2, stride: int | None = None) -> np.ndarray:
    """§4: max pooling on a (B, C, H, W) array with a k x k window and the given stride
    (default: stride = k, non-overlapping windows). No padding. Output (B, C, oh, ow) with
    oh = conv_out(H, k, stride).

    Example: a 4x4 map [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] with k=2 pools to
    [[6, 8], [14, 16]].
    """
    # TODO: default stride = k; loop over output cells; out[:, :, i, j] = window.max(axis=(2, 3))
    raise NotImplementedError("Task: max_pool2d")


# ---------------------------------------------------------------- Task 5
class SmallCNN(nn.Module):
    """§5: the universal pattern on a 1-channel image.

    block1: Conv2d(1, 16, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    block2: Conv2d(16, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    block3: Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU
    then AdaptiveAvgPool2d(1) -> flatten -> Linear(64, n_classes). Return LOGITS.

    For a (B, 1, 28, 28) batch the stages give (B,16,14,14), (B,32,7,7), (B,64,7,7),
    (B,64,1,1), (B,64), (B,n_classes). Total parameters with n_classes=10: 24,170.
    feature_shapes(x) returns the shape after each named stage as a dict.
    """

    def __init__(self, n_classes: int = 10):
        # TODO: super().__init__(); define self.block1, self.block2, self.block3 (nn.Sequential), self.gap = nn.AdaptiveAvgPool2d(1), self.head = nn.Linear(64, n_classes)
        raise NotImplementedError("Task: SmallCNN.__init__")

    def feature_shapes(self, x: torch.Tensor) -> dict:
        """{"block1": (B,16,14,14), "block2": ..., "block3": ..., "gap": ..., "logits": ...}"""
        # TODO: run x through block1, block2, block3, gap recording tuple(x.shape) under each name; then "logits" = shape of self.head(x.flatten(1))
        raise NotImplementedError("Task: SmallCNN.feature_shapes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: block1 -> block2 -> block3 -> gap -> flatten(1) -> head
        raise NotImplementedError("Task: SmallCNN.forward")


# ---------------------------------------------------------------- Task 6
def train_stripes_cnn(n_train: int = 400, n_test: int = 200, epochs: int = 8, batch_size: int = 32,
                      lr: float = 3e-3, seed: int = 0) -> dict:
    """§10: train a tiny CNN on make_stripe_images and report held-out accuracy.

    - torch.manual_seed(seed); train = make_stripe_images(n_train, seed=seed),
      test = make_stripe_images(n_test, seed=seed + 1)  (a DIFFERENT seed: unseen images)
    - model: Conv2d(1, 8, 3, p=1) -> ReLU -> MaxPool2d(2) -> Conv2d(8, 16, 3, p=1) -> ReLU
      -> MaxPool2d(2) -> AdaptiveAvgPool2d(1) -> Flatten -> Linear(16, 2)
    - Adam(lr), CrossEntropyLoss (logits + int64 labels), shuffled mini-batches per epoch
    - after each epoch: model.eval() + no_grad() test accuracy via argmax

    Returns {"test_acc": float (last epoch), "train_loss": [per-epoch mean loss],
             "test_acc_curve": [per-epoch acc], "n_params": int}.
    """
    # TODO: seed; make train/test images (different seeds); build the nn.Sequential from the docstring; Adam + CrossEntropyLoss; per epoch: shuffled mini-batches, then eval()/no_grad() test accuracy; return the dict
    raise NotImplementedError("Task: train_stripes_cnn")


if __name__ == "__main__":
    print("output sizes: 28,k3 ->", conv_out(28, 3), "| 28,k3,p1 ->", conv_out(28, 3, p=1),
          "| 224,k7,s2,p3 ->", conv_out(224, 7, s=2, p=3))
    print("conv params: (1->16, k3)", conv_params(1, 16, 3), "| (64->128, k3)", conv_params(64, 128, 3))
    img, ker = make_edge_example()
    mine = cross_correlate2d(img, ker)
    ref = F.conv2d(torch.tensor(img).view(1, 1, 6, 6), torch.tensor(ker).view(1, 1, 3, 3))[0, 0].numpy()
    print("cross-correlation matches torch:", np.allclose(mine, ref), "\n", mine)
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    print("max pool 2x2:\n", max_pool2d(x, 2)[0, 0])
    net = SmallCNN()
    print("SmallCNN shapes:", net.feature_shapes(torch.randn(1, 1, 28, 28)),
          "| params:", sum(p.numel() for p in net.parameters()))
    t0 = time.time()
    res = train_stripes_cnn()
    print(f"stripes CNN ({res['n_params']} params): test acc {res['test_acc']:.3f} in {time.time() - t0:.1f}s; "
          f"curve {[round(a, 2) for a in res['test_acc_curve']]}")
