from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/pivot_stage2/v1"
DATA = ROOT / "dataset/PIVOT/stage2_v1"
SPLITS = ("train", "validation", "test")
TASKS = {"stocknet": (20, [1, 3, 5], 88), "caiso": (360, [24, 72, 168], 20), "jena": (360, [24, 72, 168, 336], 21)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=json_default).encode()).hexdigest()


def json_default(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seal(directory, metadata):
    directory = Path(directory)
    write_json(directory / "_SUCCESS.json", {**metadata, "created_at": utc_now(),
        "artifacts": {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*"))
                      if p.is_file() and p.name != "_SUCCESS.json"}})


def verify(directory):
    directory = Path(directory)
    record = read_json(directory / "_SUCCESS.json")
    for name, expected in record["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"Artifact integrity failed: {directory}/{name}")
    return record


@contextmanager
def new_directory(path):
    path = Path(path)
    resolved = path.resolve()
    for protected in (ROOT / "dataset/Fidel-TS", ROOT / "artifacts/fidel_ts", ROOT / "docs/fidel_ts"):
        if resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
            raise ValueError("Historical data/evidence is read-only")
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    except BaseException as exc:
        write_json(path / "failure.json", {"at": utc_now(), "type": type(exc).__name__, "error": str(exc)})
        raise


def versions():
    from importlib.metadata import version, PackageNotFoundError
    result = {"python": platform.python_version()}
    for key in ("torch", "numpy", "pandas", "pyarrow", "scikit-learn", "exchange-calendars", "sentence-transformers", "transformers"):
        try:
            result[key] = version(key)
        except PackageNotFoundError:
            result[key] = None
    return result
