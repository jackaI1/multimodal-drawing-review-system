# C++ pixel comparison

Implement the supplied C++ skill demonstration in the public drawing-review
repository. The repeated shifted-XOR comparison in symbol detection and moved
content suppression is the useful hot loop: each candidate shift currently
allocates rolled arrays and a pixel-difference array.

Use a C++17 pybind11 kernel to return the minimum differing-pixel count over
inclusive shifts `[-radius, radius]` on each axis, with NumPy roll's circular
boundary semantics. Preserve all density thresholds, padding, verdicts and
region logic. A Python/NumPy implementation retains source-only operation.

## Approaches

- Chosen implementation: scikit-build-core and CMake build a small extension
  during `pip install .`; a Python dispatcher selects it when available. This
  keeps the existing NumPy interface and avoids rolling temporary arrays.
- Accelerating density reduction instead would touch float32 reduction ordering
  and risks changing threshold decisions. It does not remove the nested shift
  loop, so it is excluded.
- Adding a standalone raw RGB comparison would not help the current pipeline.

The build system and kernel boundary are implementation choices made by the
agent under the supplied task. Inputs are equally shaped 2D boolean masks;
strided/read-only arrays work through contiguous normalization. Empty masks
return zero, invalid masks/radii raise, and huge radii are reduced to equivalent
circular shifts. Native work releases the GIL and uses one thread.

Tests compare exact integer results against the original exhaustive NumPy loop,
including random, shifted, edited, empty, strided and tiny images. Integration
tests exercise both consumers and source-only fallback. CI builds and tests on
Ubuntu and macOS, also checking an installed wheel outside the source tree.

Benchmark deterministic synthetic drawing masks with real edits, warm both
backends, assert parity, and report median call time and speedup. Time includes
Python dispatch and any input normalization; exclude generation and compilation.
Document machine, sizes and repetitions, and distinguish kernel speedup from
whole-pipeline speedup. No client or application records enter this work.

The supplied task authorizes a branch, public PR, and merge after CI passes.
