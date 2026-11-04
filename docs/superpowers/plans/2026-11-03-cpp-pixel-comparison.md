# C++ Pixel Comparison Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate existing drawing comparisons with identical C++17 results and a Python fallback.

**Architecture:** Share a shifted-XOR minimum kernel between symbol detection and moved-content matching. Keep NumPy fallback; compile pybind11 with scikit-build-core/CMake.

**Tech Stack:** CPython 3.12, NumPy, C++17, pybind11, CMake, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-03-cpp-pixel-comparison-design.md`

## Global Constraints

- Preserve existing results and run the existing tests plus parity tests.
- Use synthetic data only; never commit private data or secrets.
- Work on `skills/cpp-core`; open a PR and merge after CI passes.
- Keep the repository public and record a measured benchmark in README.

## Review Focus

- Installed-wheel imports must use the compiled module, despite source-tree shadowing.
- Non-contiguous and read-only masks must compare identically.
- Empty or tiny shapes and radii larger than dimensions must preserve circular semantics.
- A missing extension must permit full Python operation, with explicit native selection failing clearly.
- Both original consumers must retain comparison results and pipeline outputs.

### Task 1: Shared kernel, build, fallback and integration

**Files:** Create `redlinecheck/_pixels.py`, `cpp/pixels.cpp`, `CMakeLists.txt`, `pyproject.toml`, `tests/test_pixels.py`; modify `redlinecheck/diff.py`.

**Interfaces:** Produce `minimum_shifted_xor(base, final, radius, *, backend="auto") -> int` with `python`, `cpp`, `auto` selection. Consume boolean masks already used by the diff pipeline.

- [ ] Write reference/parity, validation, fallback, installed backend and consumer tests.
- [ ] Run `.venv/bin/python -m pytest tests/test_pixels.py -q`; expect failure because `_pixels` does not exist.
- [ ] Implement Python dispatcher/reference, C++ kernel, build configuration, and both consumer calls.
- [ ] Build editable for local testing; run all tests with native and Python backends. Expect exact parity and the existing suite to pass.
- [ ] Commit the implementation.

### Task 2: Reproducible evidence and public CI

**Files:** Create `benchmarks/benchmark_pixels.py`, `.github/workflows/tests.yml`; update `README.md` and `.gitignore`.

**Interfaces:** Consume Task 1's backend selector. Produce deterministic benchmark output and CI for native/fallback installs on Ubuntu and macOS.

- [ ] Add benchmark smoke coverage; run it before implementation and expect the script to be missing.
- [ ] Implement deterministic synthetic drawing pairs, parity assertions, warmup and median timings.
- [ ] Build/install wheel outside the checkout, verify the native import and run benchmark.
- [ ] Add CI installation, full tests, native import assertion and Python-only build/test coverage.
- [ ] Document build/fallback instructions, skills/file paths and measured speedup.
- [ ] Run all tests and benchmark, review the full diff, commit, open PR, and merge after CI passes.
