#!/usr/bin/env python
"""Thin wrapper so you can run `python comparator.py A.pdf B.pdf -o report`.

All logic lives in the `redlinecheck` package; see its README.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Leave a CPU core free so the comparison doesn't make the machine unusable.
from redlinecheck._compute import limit_threads
limit_threads(leave_free=1)

from redlinecheck.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
