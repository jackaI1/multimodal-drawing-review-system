"""Exact shifted boolean-mask comparison, with optional C++ acceleration."""
from __future__ import annotations

import operator
import os

import numpy as np

try:
    from . import _pixels_cpp as _native
except ImportError:
    _native = None

NATIVE_AVAILABLE = _native is not None


def minimum_shifted_xor(base: np.ndarray, final: np.ndarray, radius: int,
                        *, backend: str = "auto") -> int:
    """Minimum XOR count over circular shifts within ``+/-radius`` on both axes.

    Accept equally shaped 2D boolean masks (including strided/read-only views).
    ``auto`` uses C++ when built, otherwise the Python/NumPy implementation.
    REDLINECHECK_PIXEL_BACKEND can override ``auto`` for reproducible comparisons.
    Neither input is modified. Empty inputs have zero differing pixels.
    """
    base, final = np.asarray(base), np.asarray(final)
    if base.ndim != 2 or final.ndim != 2 or base.shape != final.shape:
        raise ValueError("masks must be equally shaped 2D arrays")
    if base.dtype != np.bool_ or final.dtype != np.bool_:
        raise TypeError("masks must have boolean dtype")
    radius = operator.index(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if backend == "auto":
        backend = os.environ.get("REDLINECHECK_PIXEL_BACKEND", "auto")
    if backend not in ("auto", "python", "cpp"):
        raise ValueError("backend must be auto, python or cpp")
    if backend == "cpp" and _native is None:
        raise RuntimeError("C++ extension not built; install with pip install .")
    # Larger circular shifts only repeat residues already in this search.
    # Clamp in Python before passing to C++ so even huge Python integers work.
    radius = min(radius, max(base.shape) // 2)
    if _native is not None and backend != "python":
        return _native.minimum_shifted_xor(
            np.ascontiguousarray(base), np.ascontiguousarray(final), radius)
    return _minimum_shifted_xor_python(base, final, radius)


def _minimum_shifted_xor_python(base, final, radius):
    """Original NumPy roll/XOR search; no compiled extension required."""
    if not base.size:
        return 0
    best = int(np.count_nonzero(base ^ final))
    ry, rx = min(radius, base.shape[0] // 2), min(radius, base.shape[1] // 2)
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if dy or dx:
                best = min(best, int(np.count_nonzero(
                    base ^ np.roll(np.roll(final, dy, 0), dx, 1))))
    return best
