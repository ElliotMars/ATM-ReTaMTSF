#!/usr/bin/env python3
"""Prepare independent N1 or N2 artifacts; no Dataset or training implementation."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    directory = Path(__file__).resolve().parent
    sys.path[:] = [str(directory.parents[1])] + [p for p in sys.path if Path(p or ".").resolve() != directory]

from scripts.fidel_ts.numerical_cli import run_numerical

if __name__ == "__main__":
    raise SystemExit(run_numerical("prepare"))
