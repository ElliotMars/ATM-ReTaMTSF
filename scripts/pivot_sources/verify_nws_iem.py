#!/usr/bin/env python3
"""Manual NWS/IEM probe, audit and coverage CLI."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pivot_sources.cli import run
if __name__ == "__main__":
    raise SystemExit(run("nws"))
