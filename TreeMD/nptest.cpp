#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <cblas.h>
#include <chrono>
#include <iostream>

namespace py = pybind11;

template <typename T>
py::tuple matmul_blas(py::array_t<T> A, py::array_t<T> B) {
    py::buffer_info a_buf = A.request();
    py::buffer_info b_buf = B.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2)
        throw std::runtime_error("Input arrays must be 2D");

    const int n = a_buf.shape[0];
    const int m = a_buf.shape[1];
    const int p = b_buf.shape[1];

    if (b_buf.shape[0] != m)
        throw std::runtime_error("Inner dimensions must match");

    auto C = py::array_t<T>({n, p});
    py::buffer_info c_buf = C.request();

    auto *a_ptr = static_cast<T *>(a_buf.ptr);
    auto *b_ptr = static_cast<T *>(b_buf.ptr);
    auto *c_ptr = static_cast<T *>(c_buf.ptr);

    const T alpha = 1.0;
    const T beta = 0.0;

    auto t1 = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_same<T, float>::value) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, p, m, alpha, a_ptr, m, b_ptr, p, beta, c_ptr, p);
    } else if constexpr (std::is_same<T, double>::value) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, p, m, alpha, a_ptr, m, b_ptr, p, beta, c_ptr, p);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t2 - t1).count();

    return py::make_tuple(C, elapsed);
}

PYBIND11_MODULE(nptest, m) {
    m.def("matmul_f32", &matmul_blas<float>, "Matrix multiply (float32, via BLAS)");
    m.def("matmul_f64", &matmul_blas<double>, "Matrix multiply (float64, via BLAS)");
}
