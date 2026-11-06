from pivot_stage2 import suite
from pivot_stage2.common import write_json
from pivot_stage2.configurations import configuration


def test_formal_plan_only_admits_selected_source_and_three_fixed_seeds(tmp_path, monkeypatch):
    write_json(tmp_path / "admission.json", {"datasets": {
        "stocknet": {"main_enabled": True, "numeric_enabled": True},
        "caiso": {"main_enabled": False, "numeric_enabled": True},
        "jena": {"main_enabled": False, "numeric_enabled": True}}})
    monkeypatch.setattr(suite, "ARTIFACTS", tmp_path)
    jobs = suite.plans("main")
    assert len(jobs) == 81 and {job["dataset"] for job in jobs} == {"stocknet"}
    assert {job["seed"] for job in jobs} == {2024, 2025, 2026}
    assert all(not job["smoke"] for job in jobs)
    pilot = suite.plans("pilot")
    assert len(pilot) == 55 and {job["seed"] for job in pilot} == {2024}
    assert all(not job["formal_training"] for job in pilot)


def test_no_go_text_configs_disabled_and_core_ablations_independent():
    assert not configuration("jena", 336, "full_pivot")["enabled"]
    assert configuration("jena", 336, "backbone")["enabled"]
    assert not configuration("stocknet", 5, "without_ctsr")["model"]["use_ctsr"]
    assert configuration("stocknet", 5, "without_temporal")["retrieval"]["strategy"] == "semantic"
    assert configuration("stocknet", 5, "without_str")["retrieval"]["strategy"] == "mean"
