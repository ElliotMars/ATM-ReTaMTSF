import numpy as np
import pytest
import torch

from pivot_stage2.models import CTSR, Fusion, build_model
from pivot_stage2.metrics import masked_mse, metrics, Accumulator, block_bootstrap


@pytest.mark.parametrize("horizon", [1, 3, 5, 24, 72, 168, 336])
@pytest.mark.parametrize("channels", [3, 20, 21, 88])
@pytest.mark.parametrize("name", ["dlinear", "patchtst", "pivot"])
def test_all_planned_horizons_mimo_forward_backward_batch_one(horizon, channels, name):
    model = build_model({"name": name, "lookback": 20, "horizon": horizon, "channels": channels,
                         "dim": 16, "heads": 4, "dropout": 0.})
    x = torch.randn(1, 20, channels)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, -1] = False
    text = torch.randn(1, channels, 2, 384)
    text_mask = torch.zeros((1, channels, 2), dtype=torch.bool)
    if name == "pivot":
        text_mask[:, 0, 0] = True
    output = model(x, mask, text, text_mask) if name != "dlinear" else model(x, mask)
    assert output.shape == (1, horizon, channels) and torch.isfinite(output).all()
    loss = masked_mse(output, torch.zeros_like(output), torch.ones_like(output, dtype=torch.bool))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ctsr_excludes_self_missing_channel_and_breaks_ties_stably():
    module = CTSR(8, 2)
    x, mask = torch.ones(1, 10, 4), torch.ones(1, 10, 4, dtype=torch.bool)
    mask[..., 3] = False
    ids, valid = module.select(x, mask)
    assert ids[0, 0].tolist() == [1, 2]
    assert not valid[0, 3].any()
    for c in range(3):
        assert c not in ids[0, c][valid[0, c]].tolist()
        assert 3 not in ids[0, c][valid[0, c]].tolist()


def test_fusion_missing_text_is_exact_identity_and_padding_is_ignored():
    module = Fusion(16).eval()
    numeric = torch.randn(1, 3, 16)
    text = torch.full((1, 3, 2, 384), float("nan"))
    mask = torch.zeros((1, 3, 2), dtype=torch.bool)
    assert torch.equal(module(numeric, text, mask), numeric)
    text[:, 0, 0] = 1
    mask[:, 0, 0] = True
    output = module(numeric, text, mask)
    assert torch.isfinite(output).all() and torch.equal(output[:, 1:], numeric[:, 1:])


def test_future_labels_are_not_model_or_ctsr_features():
    model = build_model({"lookback": 20, "horizon": 3, "channels": 3, "dim": 16, "dropout": 0.}).eval()
    x, mask = torch.randn(1, 20, 3), torch.ones(1, 20, 3, dtype=torch.bool)
    a = model(x_num=x, x_mask=mask, y=torch.zeros(1, 3, 3))
    b = model(x_num=x, x_mask=mask, y=torch.full((1, 3, 3), 1e10))
    assert torch.equal(a, b)


def test_invalid_labels_excluded_from_loss_and_raw_metrics():
    prediction = torch.tensor([[[2., 999.], [3., -999.]]], requires_grad=True)
    target = torch.tensor([[[1., float("nan")], [1., float("inf")]]])
    mask = torch.isfinite(target)
    loss = masked_mse(prediction, target, mask)
    assert loss.item() == 2.5
    loss.backward()
    assert prediction.grad[~mask].eq(0).all()
    result = metrics(prediction.detach().numpy(), target.numpy(), mask.numpy())
    assert result["mse"] == 2.5 and result["valid_label_count"] == 2


def test_metrics_streaming_rank_ic_and_block_bootstrap_are_repeatable():
    p = np.arange(60, dtype=float).reshape(2, 3, 10)
    y, mask = p + 1, np.ones_like(p, bool)
    a = Accumulator(10, True)
    a.update(p[:1], y[:1], mask[:1])
    a.update(p[1:], y[1:], mask[1:])
    assert a.result()["mse"] == 1 and a.result()["rank_ic"] == pytest.approx(1)
    assert block_bootstrap(np.arange(12), 3, draws=30) == block_bootstrap(np.arange(12), 3, draws=30)


def test_patch_encoder_does_not_drop_most_recent_samples():
    model = build_model({"name": "patchtst", "lookback": 20, "horizon": 1, "channels": 3, "dropout": 0.}).eval()
    x = torch.randn(1, 20, 3, requires_grad=True)
    y = model(x, torch.ones_like(x, dtype=torch.bool))
    y.sum().backward()
    assert x.grad[:, -1].abs().sum() > 0
