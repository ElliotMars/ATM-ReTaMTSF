"""Local provenance, immutable bundles and evidence-bound request contracts."""

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/pivot_sources/stage1_6_sources.json"
OUTPUT_NAMES = ("source_inventory.json", "sample_records.jsonl", "timestamp_semantics.json",
                "asof_policy_audit.json", "availability_scenarios.json", "coverage_probe.json",
                "required_download_estimate.json", "known_risks.json", "source_readiness.json")


class SourceError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(body):
    return hashlib.sha256(body).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path=CONFIG):
    config = read_json(path)
    if config.get("schema_version") != 1 or config.get("stage") != "1.6":
        raise SourceError("Invalid Stage 1.6 config")
    for source, channels, lookback, horizons in (("nws", 20, 360, [24, 72, 168]),
                                                ("gfs", 21, 360, [24, 72, 168, 336]),
                                                ("stocknet", 88, 20, [1, 3, 5])):
        spec = config["sources"][source]
        if (spec["channels"], spec["lookback"], spec["horizons"]) != (channels, lookback, horizons):
            raise SourceError("Frozen Stage 1.6 full-channel task changed")
    if config["sources"]["gfs"]["lag_hours"] != [3, 6, 12]:
        raise SourceError("GFS sensitivity scenarios must retain 3/6/12 hours")
    if config["sources"]["stocknet"]["split_boundaries"] != ["2015-07-01", "2015-10-01", "2016-01-01"]:
        raise SourceError("StockNet candidate split changed")
    return config


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise SourceError("Path escapes bundle")
    return path


def protected_output(path, inputs=()):
    path = Path(path).resolve()
    protected = [ROOT / "dataset", ROOT / "artifacts/fidel_ts", ROOT / "docs/fidel_ts", *inputs]
    for item in protected:
        other = Path(item).resolve()
        if path.is_relative_to(other) or other.is_relative_to(path):
            raise SourceError(f"Output overlaps a protected input: {other}")


@contextmanager
def output_run(path, binding, resume=False, inputs=()):
    path = Path(path)
    protected_output(path, inputs)
    if resume:
        if not path.is_dir() or (path / "_SUCCESS.json").exists() or (path / "_INCOMPLETE.json").exists():
            raise SourceError("Only an interrupted, unsealed probe can resume")
        if read_json(path / "run_binding.json") != binding:
            raise SourceError("Resume source/config/request plan differs")
        if any((path / name).exists() for name in OUTPUT_NAMES):
            raise SourceError("Interrupted parsing must be retried into a new directory; raw requests remain reusable")
    else:
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "run_binding.json", binding)
    try:
        yield path
    except Exception as exc:
        with (path / "failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": now(), "type": type(exc).__name__, "error": str(exc)}) + "\n")
        raise


def seal(path, binding, incomplete=False):
    artifacts = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    write_json(path / ("_INCOMPLETE.json" if incomplete else "_SUCCESS.json"),
               {**binding, "artifacts": artifacts, "sealed_at": now(),
                "meaning": "artifact integrity only; not T1 approval", "incomplete": incomplete})


def read_bundle(path):
    path = Path(path)
    marker = path / "_SUCCESS.json"
    if not marker.exists():
        marker = path / "_INCOMPLETE.json"
    result = read_json(marker)
    for name, expected in result["artifacts"].items():
        if sha256(safe_path(path, name)) != expected:
            raise SourceError(f"Bundle hash mismatch: {name}")
    return result


@lru_cache(maxsize=256)
def _quoted_file(path, expected, quote, modified_ns, size):
    return sha256(path) == expected and quote in Path(path).read_text(encoding="utf-8", errors="replace")


def evidence_ok(evidence, roles=()):
    """A human-reviewed local quote, bound to its bytes and source URL.

    This checks provenance mechanically. It cannot authenticate the human's
    semantic review; that limitation is retained in every readiness report.
    """
    if not isinstance(evidence, dict) or evidence.get("reviewed") is not True:
        return False
    if roles and evidence.get("role") not in roles:
        return False
    if not isinstance(evidence.get("source_url"), str) or not evidence["source_url"].startswith("https://") or not isinstance(evidence.get("quote"), str) or not evidence["quote"]:
        return False
    try:
        path = Path(evidence["local_file"])
        info = path.stat()
        return _quoted_file(str(path.resolve()), evidence["sha256"], evidence["quote"], info.st_mtime_ns, info.st_size)
    except (KeyError, OSError, TypeError):
        return False


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href" and value)


def discover_links(body, base):
    parser = Links()
    parser.feed(body.decode("utf-8", errors="replace"))
    return sorted({urljoin(base, link) for link in parser.urls if urlsplit(urljoin(base, link)).scheme == "https"})


def load_plan(path, source):
    if path is None:
        return None
    plan = read_json(path)
    if plan.get("source") != source or plan.get("schema_version") != 1:
        raise SourceError("Request plan source/schema mismatch")
    if not evidence_ok(plan.get("interface_evidence"), ("documented_interface",)):
        raise SourceError("Request plan requires an exact, reviewed official-documentation quote and file hash")
    if not isinstance(plan.get("requests"), list) or not plan["requests"]:
        raise SourceError("A reviewed request plan must contain explicit documented requests")
    for request in plan.get("requests", []):
        if request.get("method", "GET") not in ("GET", "HEAD"):
            raise SourceError("Only bounded GET/HEAD probes are supported")
        if not request.get("documentation_reference"):
            raise SourceError("Each exact request URL must cite its documented construction")
    return plan


def dotted(value, path, default=None):
    for part in path.split(".") if path else []:
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def mapped_records(body, contract):
    decoded = json.loads(body)
    values = dotted(decoded, contract.get("records_path", ""))
    if not isinstance(values, list):
        raise SourceError("Documented records_path did not resolve to an array; no guessed response fields")
    if not all(isinstance(value, dict) for value in values):
        raise SourceError("Source record array must contain objects")
    return values


def basic_record(dataset, source, source_id, text, url, raw_file, raw_body, **fields):
    record = dict.fromkeys(("source_time_key", "issued_at", "received_at", "run_time", "available_at",
                            "valid_start", "valid_end", "timezone", "location_id", "entity_id"))
    record.update(dataset=dataset, source=source, source_record_id=str(source_id), text=text,
                  source_url=url, source_file=str(raw_file), raw_record_sha256=digest_bytes(raw_body),
                  document_id=digest([source, str(source_id), digest_bytes(raw_body)]),
                  availability_status="unknown", availability_evidence={}, availability_policy="unknown")
    record.update(fields)
    record["document_id"] = digest([source, str(source_id), record["entity_id"], record["location_id"], record["raw_record_sha256"]])
    return record


def readiness(source_status="not_checked", temporal="not_checked", task="not_checked", evidence=(), issues=(), conditional=False):
    gates = (source_status, temporal, task)
    if any(value not in {"pass", "conditional", "blocked", "not_checked"} for value in gates):
        raise SourceError("Invalid gate status")
    decision = "NO-GO" if "blocked" in gates else "GO" if all(value == "pass" for value in gates) else "CONDITIONAL"
    return {"decision": decision, "source_status": source_status, "temporal_availability_status": temporal,
            "task_compatibility_status": task, "verified_t1_feasible": all(value == "pass" for value in gates),
            "conditional_t1_feasible": bool(conditional and task != "blocked" and source_status in {"pass", "conditional"}),
            "blocking_issues": list(issues), "evidence": list(evidence),
            "full_coverage_status": "not_yet_full_coverage_verified",
            "next_required_actions": ["review bound evidence and probe scope", "verify full-period index before large download"],
            "evidence_limit": "reviewed quotes are user assertions checked against local bytes; not independent source attestation"}
