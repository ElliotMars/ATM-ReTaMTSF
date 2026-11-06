import io
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
import pytest

from scripts.pivot_sources.common import SourceError, output_run, read_bundle, readiness, seal, sha256, write_json
from scripts.pivot_sources.http import BoundedHTTP
from scripts.pivot_sources.cli import run
from scripts.pivot_sources import cli


class Response:
    def __init__(self, body, status=200, declared=True):
        self.body, self.status = io.BytesIO(body), status
        self.headers = {"Content-Type": "application/json"}
        if declared:
            self.headers["Content-Length"] = str(len(body))
        self.read_calls = 0

    def read(self, size):
        self.read_calls += 1
        return self.body.read(size)

    def close(self):
        pass


class Opener:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def open(self, *args, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(tmp_path, responses, **overrides):
    limits = {"max_requests": 2, "max_file_bytes": 10, "max_total_bytes": 15,
              "timeout_seconds": 1, "retries": 0, "min_interval_seconds": 0}
    limits.update(overrides)
    opener = Opener(responses)
    return BoundedHTTP(tmp_path, {"example.org"}, limits, True, opener, sleeper=lambda _: None), opener


def test_offline_client_and_cli_dry_runs_do_not_touch_network_or_outputs(tmp_path, capsys):
    value, opener = client(tmp_path, [])
    value.allow_network = False
    with pytest.raises(SourceError, match="Network disabled"):
        value.fetch("https://example.org/a")
    assert opener.calls == 0
    for source in ("nws", "gfs", "stocknet"):
        output = tmp_path / source
        assert run(source, ["--mode", "dry-run", "--output-dir", str(output)]) == 0
        assert not output.exists()


def test_request_cap_includes_attempts_and_duplicate_url_is_reused(tmp_path):
    value, opener = client(tmp_path, [Response(b"{}")], max_requests=1)
    first = value.fetch("https://example.org/a")
    assert value.fetch("https://example.org/a")[1]["sha256"] == first[1]["sha256"]
    assert opener.calls == 1
    with pytest.raises(SourceError, match="Maximum HTTP"):
        value.fetch("https://example.org/b")


def test_file_limit_rejects_content_length_before_body(tmp_path):
    response = Response(b"a" * 11)
    value, _ = client(tmp_path, [response])
    with pytest.raises(SourceError, match="Content-Length exceeds"):
        value.fetch("https://example.org/a")
    assert response.read_calls == 0 and not value.state["responses"]


def test_total_budget_and_undeclared_oversized_body(tmp_path):
    value, _ = client(tmp_path, [Response(b"a" * 10), Response(b"b" * 6)])
    value.fetch("https://example.org/a")
    with pytest.raises(SourceError, match="Content-Length exceeds"):
        value.fetch("https://example.org/b")
    assert value.state["bytes_received"] == 10


def test_failed_http_response_is_not_successful_and_retry_counts(tmp_path):
    error = HTTPError("https://example.org/a", 503, "fixture", {}, None)
    value, opener = client(tmp_path, [error, Response(b"{}")], retries=1)
    value.fetch("https://example.org/a")
    assert opener.calls == 2 and value.state["attempts"][0]["status"] == "failed"
    assert len(value.state["responses"]) == 1


def test_undeclared_oversize_response_stops_at_cap_without_success(tmp_path):
    value, _ = client(tmp_path, [Response(b"a" * 11, declared=False)])
    with pytest.raises(SourceError, match="byte cap"):
        value.fetch("https://example.org/a")
    assert value.state["bytes_received"] == 10
    assert not value.state["responses"]


def test_failed_404_has_no_success_response(tmp_path):
    value, _ = client(tmp_path, [HTTPError("https://example.org/a", 404, "fixture", {}, None)])
    with pytest.raises(SourceError):
        value.fetch("https://example.org/a")
    assert not value.state["responses"] and not (tmp_path / "_SUCCESS.json").exists()


def test_sealed_outputs_hashes_and_non_overwrite(tmp_path):
    directory = tmp_path / "bundle"
    with output_run(directory, {"source": "fixture"}) as output:
        write_json(output / "data.json", {"value": 1})
        seal(output, {"source": "fixture"})
    assert read_bundle(directory)["source"] == "fixture"
    with pytest.raises(FileExistsError):
        with output_run(directory, {"source": "fixture"}):
            pass
    with pytest.raises(SourceError, match="sealed"):
        with output_run(directory, {"source": "fixture"}, resume=True):
            pass
    (directory / "data.json").write_text("{}")
    with pytest.raises(SourceError, match="hash mismatch"):
        read_bundle(directory)


def test_interrupted_probe_can_resume_without_overwriting_raw(tmp_path):
    directory = tmp_path / "resume"
    binding = {"source": "fixture"}
    with pytest.raises(RuntimeError):
        with output_run(directory, binding) as output:
            value, _ = client(output, [Response(b"{}")])
            _, response = value.fetch("https://example.org/a")
            original = sha256(output / response["raw_file"])
            raise RuntimeError("synthetic interruption")
    with output_run(directory, binding, resume=True) as output:
        value, opener = client(output, [])
        _, response = value.fetch("https://example.org/a")
        assert opener.calls == 0 and sha256(output / response["raw_file"]) == original


def test_raw_input_location_is_protected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    original = raw / "fixture.txt"
    original.write_text("unchanged")
    before = sha256(original)
    with pytest.raises(SourceError, match="protected input"):
        with output_run(raw / "new", {}, inputs=[raw]):
            pass
    assert sha256(original) == before


def test_not_checked_gates_never_become_pass_or_verified_t1():
    result = readiness("conditional", "not_checked", "not_checked")
    assert result["decision"] == "CONDITIONAL" and not result["verified_t1_feasible"]
    assert result["full_coverage_status"] == "not_yet_full_coverage_verified"
    assert readiness("conditional", "conditional", "blocked")["decision"] == "NO-GO"


def test_failed_probe_emits_incomplete_decision_without_success(tmp_path, monkeypatch):
    fake = SimpleNamespace(state={"requests_used": 1, "bytes_received": 0, "bytes_budget_charged": 0})
    monkeypatch.setattr(cli, "BoundedHTTP", lambda *args, **kwargs: fake)

    def denied(*args):
        raise SourceError("HTTP 403 synthetic fixture")

    monkeypatch.setattr(cli, "planned_probe", denied)
    output = tmp_path / "failed_probe"
    assert run("nws", ["--mode", "probe", "--allow-network", "--output-dir", str(output)]) == 2
    assert not (output / "_SUCCESS.json").exists()
    assert (output / "_INCOMPLETE.json").exists()
    assert read_bundle(output)["incomplete"]
    from scripts.pivot_sources.common import read_json
    assert read_json(output / "source_readiness.json")["decision"] == "NO-GO"
