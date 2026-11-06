import pytest

from pivot_stage2.common import read_json, verify
from pivot_stage2.train import run
from pivot_stage2.weather import iem_url


def test_single_optimizer_step_checkpoint_resume_and_evaluation(bundle_factory, tmp_path):
    directory = bundle_factory(channels=3)
    config = {"dataset": "synthetic", "data_directory": str(directory), "seed": 2024, "horizon": 3,
              "device": "cpu", "model": {"name": "pivot", "dim": 16, "heads": 4, "dropout": 0.},
              "retrieval": {"strategy": "semantic_temporal", "top_k": 2, "lag_hours": 1}}
    first = tmp_path / "smoke1"
    result = run(config, first, smoke=True)
    assert result["optimizer_steps"] == 1 and result["checkpoint_roundtrip"] and result["no_text_finite"]
    assert result["not_a_research_result"] and result["test"]["samples"] == 1
    verify(first)
    resumed = run(config, tmp_path / "smoke2", smoke=True, resume=first / "latest.pt")
    assert resumed["optimizer_steps"] == 1
    assert len(read_json(tmp_path / "smoke2/history.json")) == 2


def test_iem_documented_query_is_bounded_and_uses_real_parameters():
    url = iem_url("AFDMTR", "2021-07-15T00:00:00Z", "2021-07-16T00:00:00Z")
    assert "pil=AFDMTR" in url and "fmt=zip" in url and "order=asc" in url
    with pytest.raises(ValueError):
        iem_url("AFDMTR", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z")


def test_no_overwrite_successful_training_output(bundle_factory, tmp_path):
    config = {"dataset": "synthetic", "data_directory": str(bundle_factory()), "seed": 2024, "horizon": 1,
              "model": {"name": "zero"}, "retrieval": {"strategy": "none"}}
    path = tmp_path / "no_overwrite"
    run(config, path, smoke=True)
    with pytest.raises(FileExistsError):
        run(config, path, smoke=True)
