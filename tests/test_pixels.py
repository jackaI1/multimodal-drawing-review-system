"""Exact parity with the pre-extension shifted pixel comparison."""
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from redlinecheck import _pixels as pixels
from redlinecheck import diff
from redlinecheck.config import Config
from types import SimpleNamespace


def original_minimum(base, final, radius):
    best = int(np.count_nonzero(base ^ final))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                best = min(best, int(np.count_nonzero(
                    base ^ np.roll(np.roll(final, dy, 0), dx, 1))))
    return best


@pytest.mark.parametrize("backend", ["python", "cpp", "auto"])
@pytest.mark.parametrize("shape,radius", [
    ((0, 5), 2), ((5, 0), 1), ((1, 1), 0), ((1, 9), 3),
    ((7, 1), 2), ((3, 4), 8), ((19, 27), 0), ((19, 27), 3),
])
def test_random_masks_match_original_exactly(backend, shape, radius):
    if backend == "cpp" and not pixels.NATIVE_AVAILABLE:
        pytest.skip("extension not built")
    rng = np.random.default_rng(42)
    for _ in range(5):
        base = rng.random(shape) < 0.3
        final = rng.random(shape) < 0.3
        assert pixels.minimum_shifted_xor(base, final, radius, backend=backend) == original_minimum(base, final, radius)


@pytest.mark.parametrize("backend", ["python", "cpp", "auto"])
def test_shifted_edited_strided_readonly_masks(backend):
    if backend == "cpp" and not pixels.NATIVE_AVAILABLE:
        pytest.skip("extension not built")
    base = np.zeros((28, 34), bool)
    base[2:10, 4:12] = True
    final = np.roll(base, (2, -3), axis=(0, 1))
    assert pixels.minimum_shifted_xor(base, final, 3, backend=backend) == 0
    final[20, 20] = True
    for b, f in [(base, final), (base.T, final.T),
                 (base[::-1, ::2], final[::-1, ::2])]:
        b.flags.writeable = False
        f.flags.writeable = False
        assert pixels.minimum_shifted_xor(b, f, 3, backend=backend) == original_minimum(b, f, 3)


def test_huge_radius_has_finite_circular_search():
    base = np.eye(3, 4, dtype=bool)
    final = np.roll(base, 2, axis=1)
    assert pixels.minimum_shifted_xor(base, final, 10**100) == original_minimum(base, final, 4)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_noncanonical_numpy_boolean_storage(backend):
    if backend == "cpp" and not pixels.NATIVE_AVAILABLE:
        pytest.skip("extension not built")
    base = np.array([[0, 255, 2], [128, 0, 1]], dtype=np.uint8).view(np.bool_)
    final = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8).view(np.bool_)
    assert pixels.minimum_shifted_xor(base, final, 0, backend=backend) == 0


@pytest.mark.parametrize("base,final,radius,error", [
    (np.zeros((2, 3), bool), np.zeros((3, 2), bool), 1, ValueError),
    (np.zeros(3, bool), np.zeros(3, bool), 1, ValueError),
    (np.zeros((2, 3), np.uint8), np.zeros((2, 3), bool), 1, TypeError),
    (np.zeros((2, 3), bool), np.zeros((2, 3), bool), -1, ValueError),
    (np.zeros((2, 3), bool), np.zeros((2, 3), bool), 1.5, TypeError),
])
def test_invalid_inputs_raise(base, final, radius, error):
    with pytest.raises(error):
        pixels.minimum_shifted_xor(base, final, radius)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        pixels.minimum_shifted_xor(np.zeros((1, 1), bool), np.zeros((1, 1), bool), 0, backend="typo")


def test_missing_native_falls_back(monkeypatch):
    monkeypatch.setattr(pixels, "_native", None)
    base = np.eye(7, dtype=bool)
    final = np.roll(base, 1, axis=0)
    monkeypatch.setenv("REDLINECHECK_PIXEL_BACKEND", "auto")
    assert pixels.minimum_shifted_xor(base, final, 1) == 0
    with pytest.raises(RuntimeError, match="not built"):
        pixels.minimum_shifted_xor(base, final, 1, backend="cpp")


def test_source_import_without_extension(monkeypatch):
    # Block the native module in a fresh interpreter, including import-time selection.
    monkeypatch.setenv("REDLINECHECK_PIXEL_BACKEND", "auto")
    script = """
import sys
sys.modules['redlinecheck._pixels_cpp'] = None
from redlinecheck import _pixels
import numpy as np
assert not _pixels.NATIVE_AVAILABLE
a = np.eye(5, dtype=bool)
assert _pixels.minimum_shifted_xor(a, np.roll(a, 1, 0), 1, backend='auto') == 0
"""
    subprocess.run([sys.executable, "-c", script], check=True,
                   cwd=Path(__file__).resolve().parents[1])


def test_environment_can_force_python(monkeypatch):
    class FailingNative:
        def minimum_shifted_xor(self, *args):
            raise AssertionError("native backend must not run")
    monkeypatch.setattr(pixels, "_native", FailingNative())
    monkeypatch.setenv("REDLINECHECK_PIXEL_BACKEND", "python")
    base = np.eye(6, dtype=bool)
    assert pixels.minimum_shifted_xor(base, base, 2) == 0


@pytest.mark.parametrize("edited", [False, True])
def test_moved_content_consumer_preserves_verdict(edited):
    rng = np.random.default_rng(15)
    base = rng.random((20, 23)) < 0.2
    final = np.roll(base, 1, axis=0)
    if edited:
        final[5:10, 5:15] ^= True
    radius, threshold = 2, 0.2
    pad = ((radius, radius), (radius, radius))
    expected = 2 * original_minimum(np.pad(base, pad), np.pad(final, pad), radius) / (base.sum() + final.sum()) <= threshold
    assert diff._content_matches(base, final, radius, threshold) == expected


def test_native_boundary_rejects_invalid_arrays():
    if not pixels.NATIVE_AVAILABLE:
        pytest.skip("extension not built")
    native = importlib.import_module("redlinecheck._pixels_cpp")
    base = np.zeros((2, 3), bool)
    with pytest.raises(ValueError):
        native.minimum_shifted_xor(base, base.T.copy(), 1)
    with pytest.raises(ValueError):
        native.minimum_shifted_xor(base, base, -1)
    with pytest.raises(ValueError):
        native.minimum_shifted_xor(base.ravel(), base.ravel(), 1)
    with pytest.raises(TypeError):
        native.minimum_shifted_xor(base.astype(np.uint8), base, 1)


def test_symbol_consumer_matches_original(monkeypatch):
    y, x = np.ogrid[:80, :80]
    distance = (y - 40) ** 2 + (x - 40) ** 2
    base = (distance >= 25**2) & (distance <= 27**2)
    base[30:49, 36:39] = True
    final = base.copy()
    final[42:49, 39:48] = True
    page = SimpleNamespace(base_ink=base, final_ink=final)
    cfg = Config(analysis_dpi=72, symbol_min_pt=1, symbol_max_pt=80,
                 symbol_min_change_pt=1, symbol_min_frac=0.01,
                 symbol_max_frac=2)
    actual = diff.detect_symbol_changes(page, cfg)
    assert actual, "fixture must exercise the symbol comparison"
    monkeypatch.setattr(diff, "minimum_shifted_xor", original_minimum)
    assert actual == diff.detect_symbol_changes(page, cfg)


def test_benchmark_runs_and_reports_parity():
    if not pixels.NATIVE_AVAILABLE:
        pytest.skip("benchmark requires compiled extension")
    run = subprocess.run([sys.executable, "benchmarks/benchmark_pixels.py",
                          "--size", "48", "--radius", "2", "--repeat", "2"],
                         cwd=Path(__file__).resolve().parents[1],
                         text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "identical" in run.stdout
    assert "Speedup:" in run.stdout
