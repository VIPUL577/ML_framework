#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "seera_engine.hpp"

namespace py = pybind11;
using arr_f = py::array_t<float, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<int32_t, py::array::c_style>;

PYBIND11_MODULE(seera_cpp, m) {
    m.doc() = "Seera C++ engine — OpenBLAS + OpenMP backend";

    // ── Matmul ──────────────────────────────────────────────
    m.def("matmul", [](arr_f A, arr_f B) -> arr_f {
        auto a = A.unchecked<2>();
        auto b = B.unchecked<2>();
        int M = a.shape(0), K = a.shape(1), N = b.shape(1);
        arr_f C({(ssize_t)M, (ssize_t)N});
        seera::matmul(A.data(), B.data(), C.mutable_data(), M, K, N);
        return C;
    }, "Matrix multiply: A(M,K) @ B(K,N) -> C(M,N)");

    // ── Element-wise add ────────────────────────────────────
    m.def("add", [](arr_f A, arr_f B) -> arr_f {
        if (A.size() != B.size())
            throw std::runtime_error("add: arrays must have same size");
        arr_f out(A.request().shape);
        seera::add_arrays(A.data(), B.data(), out.mutable_data(), A.size());
        return out;
    });

    // ── Element-wise mul ────────────────────────────────────
    m.def("mul", [](arr_f A, arr_f B) -> arr_f {
        if (A.size() != B.size())
            throw std::runtime_error("mul: arrays must have same size");
        arr_f out(A.request().shape);
        seera::mul_arrays(A.data(), B.data(), out.mutable_data(), A.size());
        return out;
    });

    // ── Activation helpers (return out, grad same shape) ────
    auto def_unary = [&](const char* name, auto fn) {
        m.def(name, [fn](arr_f X) -> py::tuple {
            auto buf = X.request();
            arr_f out(buf.shape);
            arr_f grad(buf.shape);
            fn(X.data(), out.mutable_data(), grad.mutable_data(), (int)X.size());
            return py::make_tuple(out, grad);
        });
    };

    def_unary("relu",    seera::relu_fwd);
    def_unary("sigmoid", seera::sigmoid_fwd);
    def_unary("tanh_act",seera::tanh_fwd);
    def_unary("log_act", seera::log_fwd);
    def_unary("exp_act", seera::exp_fwd);
    def_unary("abs_act", seera::abs_fwd);
    def_unary("sqrt_act",seera::sqrt_fwd);

    // ── Pow ─────────────────────────────────────────────────
    m.def("pow_act", [](arr_f X, float exponent) -> py::tuple {
        auto buf = X.request();
        arr_f out(buf.shape);
        arr_f grad(buf.shape);
        seera::pow_fwd(X.data(), exponent, out.mutable_data(), grad.mutable_data(), X.size());
        return py::make_tuple(out, grad);
    });

    // ── Clip ────────────────────────────────────────────────
    m.def("clip_act", [](arr_f X, float lo, float hi) -> py::tuple {
        auto buf = X.request();
        arr_f out(buf.shape);
        arr_f grad(buf.shape);
        seera::clip_fwd(X.data(), lo, hi, out.mutable_data(), grad.mutable_data(), X.size());
        return py::make_tuple(out, grad);
    });

    // ── Softmax forward ─────────────────────────────────────
    m.def("softmax", [](arr_f X) -> arr_f {
        auto buf = X.request();
        if (buf.ndim < 2) throw std::runtime_error("softmax: need >= 2D");
        int N = 1;
        for (int i = 0; i < buf.ndim - 1; i++) N *= buf.shape[i];
        int C = buf.shape[buf.ndim - 1];
        arr_f out(buf.shape);
        seera::softmax_fwd(X.data(), out.mutable_data(), N, C);
        return out;
    });

    // ── Softmax VJP backward ────────────────────────────────
    m.def("softmax_vjp", [](arr_f S, arr_f dout) -> arr_f {
        auto buf = S.request();
        int N = 1;
        for (int i = 0; i < buf.ndim - 1; i++) N *= buf.shape[i];
        int C = buf.shape[buf.ndim - 1];
        arr_f dx(buf.shape);
        seera::softmax_vjp(S.data(), dout.data(), dx.mutable_data(), N, C);
        return dx;
    });

    // ── Conv2D Forward ──────────────────────────────────────
    m.def("conv2d_forward", [](arr_f X, arr_f W, int strideh, int stridew, int padh, int padw) -> arr_f {
        auto xb = X.request();
        auto wb = W.request();
        int N = xb.shape[0], C = xb.shape[1], H = xb.shape[2], Wi = xb.shape[3];
        int F = wb.shape[0], KH = wb.shape[2], KW = wb.shape[3];
        int OH = (H + 2*padh - KH) / strideh + 1;
        int OW = (Wi + 2*padw - KW) / stridew + 1;
        arr_f out({(ssize_t)N, (ssize_t)F, (ssize_t)OH, (ssize_t)OW});
        seera::conv2d_forward(X.data(), W.data(), out.mutable_data(),
                              N, C, H, Wi, F, KH, KW, strideh, stridew, padh, padw);
        return out;
    });

    // ── Conv2D Backward ─────────────────────────────────────
    m.def("conv2d_backward", [](arr_f dout, arr_f X, arr_f W,
                                int strideh, int stridew, int padh, int padw) -> py::tuple {
        auto xb = X.request();
        auto wb = W.request();
        int N = xb.shape[0], C = xb.shape[1], H = xb.shape[2], Wi = xb.shape[3];
        int F = wb.shape[0], KH = wb.shape[2], KW = wb.shape[3];
        arr_f dX(xb.shape);
        arr_f dW(wb.shape);
        seera::conv2d_backward(dout.data(), X.data(), W.data(),
                               dX.mutable_data(), dW.mutable_data(),
                               N, C, H, Wi, F, KH, KW, strideh, stridew, padh, padw);
        return py::make_tuple(dX, dW);
    });

    // ── MaxPool2D Forward ───────────────────────────────────
    m.def("maxpool2d_forward", [](arr_f X, int KH, int KW,
                                  int strideh, int stridew, int padh, int padw) -> py::tuple {
        auto xb = X.request();
        int N = xb.shape[0], C = xb.shape[1], H = xb.shape[2], W = xb.shape[3];
        int OH = (H + 2*padh - KH) / strideh + 1;
        int OW = (W + 2*padw - KW) / stridew + 1;
        arr_f out({(ssize_t)N, (ssize_t)C, (ssize_t)OH, (ssize_t)OW});
        arr_i mask({(ssize_t)N, (ssize_t)C, (ssize_t)OH, (ssize_t)OW});
        seera::maxpool2d_forward(X.data(), out.mutable_data(), mask.mutable_data(),
                                 N, C, H, W, KH, KW, strideh, stridew, padh, padw);
        return py::make_tuple(out, mask);
    });

    // ── MaxPool2D Backward ──────────────────────────────────
    m.def("maxpool2d_backward", [](arr_f dout, arr_i mask,
                                   int N, int C, int H, int W,
                                   int KH, int KW, int strideh, int stridew, int padh, int padw) -> arr_f {
        auto db = dout.request();
        int OH = db.shape[2], OW = db.shape[3];
        arr_f dX({(ssize_t)N, (ssize_t)C, (ssize_t)H, (ssize_t)W});
        seera::maxpool2d_backward(dout.data(), mask.data(), dX.mutable_data(),
                                  N, C, H, W, OH, OW, KH, KW, strideh, stridew, padh, padw);
        return dX;
    });

    // ── Upsample Forward ────────────────────────────────────
    m.def("unpooling_forward", [](arr_f X, int sh, int sw) -> arr_f {
        auto xb = X.request();
        int N = xb.shape[0], C = xb.shape[1], H = xb.shape[2], W = xb.shape[3];
        arr_f out({(ssize_t)N, (ssize_t)C, (ssize_t)(H*sh), (ssize_t)(W*sw)});
        seera::unpooling_fwd(X.data(), out.mutable_data(), N, C, H, W, sh, sw);
        return out;
    });

    // ── Upsample Backward ───────────────────────────────────
    m.def("unpooling_backward", [](arr_f dout, int N, int C, int H, int W,
                                  int sh, int sw) -> arr_f {
        arr_f dx({(ssize_t)N, (ssize_t)C, (ssize_t)H, (ssize_t)W});
        seera::unpooling_bwd(dout.data(), dx.mutable_data(), N, C, H, W, sh, sw);
        return dx;
    });

    // ── ConvTranspose2D Forward ─────────────────────────────
    m.def("conv_transpose2d_forward", [](arr_f X, arr_f W,
                                        int strideh, int stridew, int padh, int padw) -> arr_f {
        auto xb = X.request();
        auto wb = W.request();
        int N = xb.shape[0], Cin = xb.shape[1], H = xb.shape[2], Wi = xb.shape[3];
        int Cout = wb.shape[1], KH = wb.shape[2], KW = wb.shape[3];
        int Hout = (H - 1) * strideh - 2 * padh + KH;
        int Wout = (Wi - 1) * stridew - 2 * padw + KW;
        arr_f out({(ssize_t)N, (ssize_t)Cout, (ssize_t)Hout, (ssize_t)Wout});
        seera::conv_transpose2d_forward(X.data(), W.data(), out.mutable_data(),
                                        N, Cin, H, Wi, Cout, KH, KW, strideh, stridew, padh, padw);
        return out;
    });

    // ── ConvTranspose2D Backward ────────────────────────────
    m.def("conv_transpose2d_backward", [](arr_f dout, arr_f X, arr_f W,
                                         int strideh, int stridew, int padh, int padw) -> py::tuple {
        auto xb = X.request();
        auto wb = W.request();
        int N = xb.shape[0], Cin = xb.shape[1], H = xb.shape[2], Wi = xb.shape[3];
        int Cout = wb.shape[1], KH = wb.shape[2], KW = wb.shape[3];
        arr_f dX(xb.shape);
        arr_f dW(wb.shape);
        seera::conv_transpose2d_backward(dout.data(), X.data(), W.data(),
                                         dX.mutable_data(), dW.mutable_data(),
                                         N, Cin, H, Wi, Cout, KH, KW, strideh, stridew, padh, padw);
        return py::make_tuple(dX, dW);
    });

    // ── BatchNorm Forward ───────────────────────────────────
    m.def("batchnorm_forward", [](arr_f X, arr_f gamma, arr_f beta,
                                  arr_f running_mean, arr_f running_var,
                                  float momentum, float eps,
                                  bool training, bool is_2d) -> py::tuple {
        auto xb = X.request();
        int N = xb.shape[0], C = xb.shape[1];
        int H = xb.ndim > 2 ? xb.shape[2] : 1;
        int W = xb.ndim > 3 ? xb.shape[3] : 1;

        arr_f out(xb.shape);
        arr_f x_hat(xb.shape);
        arr_f mean_out({(ssize_t)C});
        arr_f std_inv_out({(ssize_t)C});

        // Get mutable pointers for running stats (in-place update)
        auto rm = running_mean.mutable_data();
        auto rv = running_var.mutable_data();

        seera::batchnorm_forward(X.data(), gamma.data(), beta.data(),
                                 rm, rv,
                                 out.mutable_data(), x_hat.mutable_data(),
                                 mean_out.mutable_data(), std_inv_out.mutable_data(),
                                 N, C, H, W, momentum, eps, training, is_2d);
        return py::make_tuple(out, x_hat, std_inv_out);
    });

    // ── BatchNorm Backward ──────────────────────────────────
    m.def("batchnorm_backward", [](arr_f dout, arr_f x_hat, arr_f std_inv,
                                   arr_f gamma, int M, bool is_2d) -> py::tuple {
        auto db = dout.request();
        int N = db.shape[0], C = db.shape[1];
        int H = db.ndim > 2 ? db.shape[2] : 1;
        int W = db.ndim > 3 ? db.shape[3] : 1;

        arr_f dx(db.shape);
        arr_f dgamma({(ssize_t)C});
        arr_f dbeta({(ssize_t)C});

        seera::batchnorm_backward(dout.data(), x_hat.data(), std_inv.data(),
                                  gamma.data(),
                                  dx.mutable_data(), dgamma.mutable_data(),
                                  dbeta.mutable_data(),
                                  N, C, H, W, M, is_2d);
        return py::make_tuple(dx, dgamma, dbeta);
    });
}
