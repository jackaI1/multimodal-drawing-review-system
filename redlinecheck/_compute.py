"""Compute politeness: cap the math-library thread pools so a big comparison doesn't
grab every CPU core and make the machine unusable while the engineer works.

MUST be imported BEFORE numpy / scipy / PyMuPDF (the BLAS/OMP thread counts are read
once, at import time). Call `limit_threads()` at the very top of an entry point.
"""
from __future__ import annotations

import os


def limit_threads(leave_free: int = 1) -> None:
    """Leave `leave_free` CPU cores free for the rest of the system. setdefault so an
    explicit env var from the user always wins."""
    try:
        cores = os.cpu_count() or 0
    except Exception:
        cores = 0
    if cores <= 0:
        return
    n = max(1, cores - max(0, leave_free))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))
