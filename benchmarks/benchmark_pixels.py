"""Compare real Python/NumPy and C++ shifted-XOR paths on synthetic drawings.

Run from the repository root after `pip install -e .`:
    python benchmarks/benchmark_pixels.py --size 512 --radius 3 --repeat 9
"""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np

# Support a source checkout as well as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redlinecheck._pixels import NATIVE_AVAILABLE, minimum_shifted_xor


def synthetic_pair(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded linework, shifted and edited so a perfect-match shortcut cannot win."""
    base = np.zeros((size, size), dtype=bool)
    gap = max(8, size // 16)
    base[gap:-gap:gap, gap:-gap] = True
    base[gap:-gap, gap:-gap:gap] = True
    rng = np.random.default_rng(42)
    for _ in range(40):
        y, x = rng.integers(0, size - 4, size=2)
        base[y:y + 3, x:x + 4] = True
    final = np.roll(base, (1, -2), axis=(0, 1))
    final[size // 3:size // 3 + 4, size // 2:size // 2 + 6] ^= True
    return base, final


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive, default=512)
    parser.add_argument("--radius", type=positive, default=3)
    parser.add_argument("--repeat", type=positive, default=9)
    args = parser.parse_args()
    if args.size < 16:
        parser.error("--size must be at least 16")
    if not NATIVE_AVAILABLE:
        parser.error("C++ extension not built; run pip install -e . first")
    base, final = synthetic_pair(args.size)
    counts = {backend: minimum_shifted_xor(base, final, args.radius, backend=backend)
              for backend in ("python", "cpp")}
    assert counts["python"] == counts["cpp"], counts
    assert counts["cpp"] > 0, "benchmark pair must contain a genuine edit"

    samples = {"python": [], "cpp": []}
    # Each backend was warmed above; alternate timing order to reduce order bias.
    for i in range(args.repeat):
        order = ("python", "cpp") if i % 2 == 0 else ("cpp", "python")
        for backend in order:
            start = perf_counter()
            count = minimum_shifted_xor(base, final, args.radius, backend=backend)
            samples[backend].append(perf_counter() - start)
            assert count == counts["python"]
    times = {key: statistics.median(values) for key, values in samples.items()}
    print(f"{platform.system()} {platform.release()} {platform.machine()}, "
          f"Python {platform.python_version()}, NumPy {np.__version__}")
    print(f"Synthetic {args.size}x{args.size} boolean drawing masks, "
          f"radius {args.radius}, {args.repeat} repetitions (median)")
    print(f"Results: identical ({counts['cpp']} differing pixels after best shift)")
    print(f"Python/NumPy: {times['python'] * 1000:.3f} ms")
    print(f"C++17:       {times['cpp'] * 1000:.3f} ms")
    print(f"Speedup: {times['python'] / times['cpp']:.2f}x (comparison kernel only)")


if __name__ == "__main__":
    main()
