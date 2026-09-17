import math
import os
import time

import pytest
import torch
import torch.nn as nn

from labs import ch16_pytorch_training as lab


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- Task 1
def test_task1_build_mlp_shapes_and_parameter_count():
    torch.manual_seed(0)
    m = lab.build_mlp(2, (32, 16), 1)
    assert isinstance(m, nn.Module)
    assert _n_params(m) == 2 * 32 + 32 + 32 * 16 + 16 + 16 * 1 + 1
    out = m(torch.randn(7, 2))
    assert tuple(out.shape) == (7, 1)
    m3 = lab.build_mlp(5, (8,), 3)
    assert tuple(m3(torch.randn(4, 5)).shape) == (4, 3)


def test_task1_outputs_are_logits_not_probabilities():
    # classic mistake: a trailing Sigmoid. Logits must be able to go outside (0, 1).
    torch.manual_seed(0)
    m = lab.build_mlp(3, (16,), 1)
    out = m(torch.randn(500, 3) * 10)
    assert (out < 0).any() or (out > 1).any()
    assert isinstance(list(m.modules())[-1], nn.Linear)


def test_task1_dropout_is_present_only_when_requested():
    assert not any(isinstance(x, nn.Dropout) for x in lab.build_mlp(2, (8,), 1).modules())
    md = lab.build_mlp(2, (8, 8), 1, dropout=0.5)
    drops = [x for x in md.modules() if isinstance(x, nn.Dropout)]
    assert len(drops) == 2 and drops[0].p == pytest.approx(0.5)


# ---------------------------------------------------------------- Task 2
def test_task2_val_loss_decreases_and_history_is_consistent():
    X_tr, X_va, y_tr, y_va = lab.make_fraud_data(seed=0)
    torch.manual_seed(0)
    model = lab.build_mlp(2, (32, 32), 1)
    hist = lab.train_with_early_stopping(model, X_tr, y_tr, X_va, y_va, lr=1e-2, max_epochs=30, patience=5)
    assert set(hist) >= {"train_loss", "val_loss", "best_epoch", "epochs_run"}
    assert len(hist["train_loss"]) == len(hist["val_loss"]) == hist["epochs_run"] <= 30
    assert min(hist["val_loss"]) < hist["val_loss"][0] - 0.1
    assert min(hist["val_loss"]) < 0.25
    assert hist["best_epoch"] == hist["val_loss"].index(min(hist["val_loss"])) + 1
    with torch.no_grad():
        acc = ((model(X_va) > 0).float() == y_va).float().mean().item()
    assert acc > 0.9


def test_task2_best_state_is_restored_and_model_is_in_eval_mode():
    # classic mistake: returning the LAST epoch's weights instead of the best ones
    X_tr, X_va, y_tr, y_va = lab.make_fraud_data(seed=1)
    torch.manual_seed(1)
    model = lab.build_mlp(2, (32, 32), 1, dropout=0.2)
    hist = lab.train_with_early_stopping(model, X_tr, y_tr, X_va, y_va, lr=3e-2, max_epochs=30, patience=3)
    assert not model.training
    with torch.no_grad():
        val_now = nn.BCEWithLogitsLoss()(model(X_va), y_va).item()
    assert val_now == pytest.approx(min(hist["val_loss"]), abs=1e-5)


def test_task2_early_stopping_actually_stops_and_is_reproducible():
    X_tr, X_va, y_tr, y_va = lab.make_fraud_data(seed=2)
    runs = []
    for _ in range(2):
        torch.manual_seed(3)
        model = lab.build_mlp(2, (64, 64), 1)
        hist = lab.train_with_early_stopping(model, X_tr, y_tr, X_va, y_va, lr=5e-2, max_epochs=30, patience=2, seed=3)
        runs.append(hist)
    assert runs[0]["val_loss"] == pytest.approx(runs[1]["val_loss"])
    # with an aggressive lr and patience=2 the loop must stop before max_epochs
    assert runs[0]["epochs_run"] < 30
    assert runs[0]["epochs_run"] == runs[0]["best_epoch"] + 2


# ---------------------------------------------------------------- Task 3
def test_task3_adam_beats_plain_sgd_from_identical_init():
    X_tr, _, y_tr, _ = lab.make_fraud_data(seed=0)
    res = lab.compare_optimizers(X_tr, y_tr, steps=60, seed=0)
    assert set(res) == {"sgd", "adam"}
    assert res["adam"] < 0.4 < res["sgd"] < math.log(2) + 0.05
    assert res["adam"] < res["sgd"] - 0.1
    # reproducible: same seed -> same numbers
    res2 = lab.compare_optimizers(X_tr, y_tr, steps=60, seed=0)
    assert res == pytest.approx(res2)


# ---------------------------------------------------------------- Task 4
def test_task4_lr_schedule_matches_warmup_then_cosine():
    lrs = lab.lr_schedule(0.1, 3, 12)
    assert len(lrs) == 12
    expected = []
    for e in range(12):
        mult = (e + 1) / 3 if e < 3 else 0.5 * (1 + math.cos(math.pi * (e - 3) / 9))
        expected.append(0.1 * mult)
    assert lrs == pytest.approx(expected, abs=1e-6)
    assert lrs[0] < lrs[1] < lrs[2] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))
    assert 0 < lrs[-1] < 0.005


def test_task4_lr_schedule_other_lengths():
    lrs = lab.lr_schedule(0.01, 2, 8)
    assert len(lrs) == 8
    assert lrs[:2] == pytest.approx([0.005, 0.01])
    assert lrs[2] == pytest.approx(0.01)


# ---------------------------------------------------------------- Task 5
def test_task5_regularization_shrinks_the_gap():
    t0 = time.time()
    res = lab.regularization_gap(seed=0)
    assert time.time() - t0 < 25
    assert set(res) == {"plain", "regularized"}
    for k in res:
        assert set(res[k]) == {"train_loss", "val_loss", "gap"}
        assert res[k]["gap"] == pytest.approx(res[k]["val_loss"] - res[k]["train_loss"])
    plain, reg = res["plain"], res["regularized"]
    assert plain["train_loss"] < 0.1           # the plain net memorizes the tiny table
    assert plain["gap"] > 0.3
    assert reg["gap"] < 0.8 * plain["gap"]
    assert reg["val_loss"] < plain["val_loss"]


# ---------------------------------------------------------------- Task 6
def test_task6_state_dict_roundtrip_is_exact():
    torch.manual_seed(0)
    model = lab.build_mlp(2, (16, 8), 1)
    model.eval()
    path = os.path.join(lab.DATA_DIR, "ch16_test_roundtrip.pt")
    fresh = lab.save_and_reload(model, lambda: lab.build_mlp(2, (16, 8), 1), path)
    assert os.path.exists(path)
    assert fresh is not model
    assert not fresh.training
    x = torch.randn(20, 2)
    with torch.no_grad():
        assert torch.allclose(fresh(x), model(x))
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    # a model that was merely rebuilt (not loaded) gives different outputs
    torch.manual_seed(99)
    other = lab.build_mlp(2, (16, 8), 1)
    with torch.no_grad():
        assert not torch.allclose(other(x), model(x))
