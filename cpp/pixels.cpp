// Shifted XOR for registered ink masks. No temporary images or worker threads.
#include <algorithm>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using Mask = py::array_t<bool, py::array::c_style>;
using Index = py::ssize_t;

Index count_differences(const unsigned char* base, const unsigned char* final,
                        Index size) {
    Index total = 0;
    // Bounded 32-bit reductions let compilers use more SIMD lanes than an
    // array-size accumulator, without overflowing on very wide input arrays.
    for (Index offset = 0; offset < size;) {
        const Index count = std::min<Index>(size - offset, 16384);
        unsigned changed = 0;
        for (Index x = 0; x < count; ++x) {
            changed += (base[offset + x] != 0) != (final[offset + x] != 0);
        }
        total += changed;
        offset += count;
    }
    return total;
}

Index minimum_shifted_xor(const Mask& base, const Mask& final, Index radius) {
    if (base.ndim() != 2 || final.ndim() != 2 ||
        base.shape(0) != final.shape(0) || base.shape(1) != final.shape(1)) {
        throw std::invalid_argument("masks must be equally shaped 2D arrays");
    }
    if (radius < 0) {
        throw std::invalid_argument("radius must be nonnegative");
    }
    const Index h = base.shape(0), w = base.shape(1);
    if (h == 0 || w == 0) {
        return 0;
    }
    // NumPy bool arrays may contain any nonzero byte for true, e.g. uint8
    // views. Reading those bytes as C++ bool would assume canonical 0/1 values.
    const auto* b = reinterpret_cast<const unsigned char*>(base.data());
    const auto* f = reinterpret_cast<const unsigned char*>(final.data());
    const Index ry = std::min(radius, h / 2);
    const Index rx = std::min(radius, w / 2);
    Index best = base.size();

    // Python owners remain alive for this call; all Python API work stays
    // outside the GIL-free region. Inputs are read-only throughout.
    py::gil_scoped_release release;
    for (Index dy = -ry; dy <= ry; ++dy) {
        for (Index dx = -rx; dx <= rx; ++dx) {
            Index changed = 0;
            const Index start_x = dx > 0 ? w - dx : -dx;
            const Index split = w - start_x;
            for (Index y = 0; y < h; ++y) {
                Index source_y = y - dy;
                if (source_y < 0) source_y += h;
                if (source_y >= h) source_y -= h;
                const auto* br = b + y * w;
                const auto* fr = f + source_y * w;
                // Split once at the circular boundary, allowing the compiler
                // to vectorize comparisons without per-pixel modulo or copies.
                changed += count_differences(br, fr + start_x, split);
                changed += count_differences(br + split, fr, start_x);
            }
            best = std::min(best, changed);
            if (best == 0) return 0;
        }
    }
    return best;
}

PYBIND11_MODULE(_pixels_cpp, m) {
    m.doc() = "C++17 shifted boolean-mask XOR comparison";
    m.def("minimum_shifted_xor", &minimum_shifted_xor,
          py::arg("base").noconvert(), py::arg("final").noconvert(),
          py::arg("radius"));
}
