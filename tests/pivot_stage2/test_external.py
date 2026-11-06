import pytest
import torch

from pivot_stage2.external import latest_legal_forecast, GFSNumericalControl
from pivot_stage2.models import build_model


def test_gfs_controls_share_legal_run_and_never_use_future_runs_or_observations():
    records = [{"origin_kind": "model_forecast", "run_id": "past", "run_time": "2021-01-01T00:00:00Z",
                "valid_time": "2021-01-02T00:00:00Z", "values": [2., 3.]},
               {"origin_kind": "model_forecast", "run_id": "future", "run_time": "2021-01-01T12:00:00Z",
                "valid_time": "2021-01-02T00:00:00Z", "values": [99., 99.]}]
    result = latest_legal_forecast(records, "2021-01-01T08:00:00Z", ["2021-01-02T00:00:00Z"], 6)
    assert result["run_id"] == "past" and result["values"].tolist() == [[2., 3.]]
    with pytest.raises(ValueError):
        latest_legal_forecast([{**records[0], "origin_kind": "actual_future_station_observation"}], "2021-01-01T08:00:00Z", [], 6)


def test_missing_gfs_numerical_control_is_identity():
    backbone = build_model({"name": "patchtst", "lookback": 20, "horizon": 3, "channels": 21, "dim": 16, "dropout": 0.}).eval()
    model = GFSNumericalControl(backbone, 5, 21).eval()
    x, mask = torch.randn(1, 20, 21), torch.ones(1, 20, 21, dtype=torch.bool)
    result = model(x, mask, torch.full((1, 3, 5), float("nan")), torch.zeros(1, 3, 5, dtype=torch.bool))
    assert torch.equal(result, backbone(x, mask))
