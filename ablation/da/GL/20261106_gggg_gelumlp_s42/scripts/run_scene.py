#!/usr/bin/env python3
"""Run one scene for this experiment via the shared DA variant runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.da_variant_runner import main

if __name__ == "__main__":
    exp = Path(__file__).resolve().parents[1]
    sys.argv = [sys.argv[0], "--experiment", str(exp), *sys.argv[1:]]
    main()
