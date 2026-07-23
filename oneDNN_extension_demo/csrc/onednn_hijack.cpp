#include "utils.h"

namespace odnn_hijack {
at::Tensor gelu_onednn(const at::Tensor&, c10::string_view);
at::Tensor tanh_onednn(const at::Tensor&);
at::Tensor sigmoid_onednn(const at::Tensor&);
at::Tensor silu_onednn(const at::Tensor&);
at::Tensor leaky_relu_onednn(const at::Tensor&, const at::Scalar&);
at::Tensor elu_onednn(const at::Tensor&, const at::Scalar&, const at::Scalar&, const at::Scalar&);
at::Tensor hardswish_onednn(const at::Tensor&);
at::Tensor relu_onednn(const at::Tensor&);
at::Tensor softmax_onednn(const at::Tensor&, int64_t, c10::optional<at::ScalarType>);
at::Tensor log_softmax_onednn(const at::Tensor&, int64_t, c10::optional<at::ScalarType>);
at::Tensor add_onednn(const at::Tensor&, const at::Tensor&, const at::Scalar&);
at::Tensor sub_onednn(const at::Tensor&, const at::Tensor&, const at::Scalar&);
at::Tensor mul_onednn(const at::Tensor&, const at::Tensor&);
at::Tensor div_onednn(const at::Tensor&, const at::Tensor&);
at::Tensor maximum_onednn(const at::Tensor&, const at::Tensor&);
at::Tensor minimum_onednn(const at::Tensor&, const at::Tensor&);
at::Tensor clamp_onednn(const at::Tensor&, const c10::optional<at::Scalar>&, const c10::optional<at::Scalar>&);
at::Tensor matmul_onednn(const at::Tensor&, const at::Tensor&);
}

// 对非 float32 输入先转 float32 再走 oneDNN kernel（确保 kernel 永远成功）
template <typename Fn, typename... Args>
at::Tensor run_or_cast(Fn&& fn, const at::Tensor& self, Args&&... args) {
    auto out = fn(self, std::forward<Args>(args)...);
    if (out.defined()) return out;
    // oneDNN kernel 无法处理（非 float32），转为 float32 后重试
    auto self_f32 = self.to(at::kFloat);
    out = fn(self_f32, std::forward<Args>(args)...);
    return out.to(self.scalar_type());
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("gelu", [](const at::Tensor& self, c10::string_view approx) -> at::Tensor {
        return run_or_cast(odnn_hijack::gelu_onednn, self, approx);
    });
    m.impl("tanh", [](const at::Tensor& self) -> at::Tensor {
        return run_or_cast(odnn_hijack::tanh_onednn, self);
    });
    m.impl("sigmoid", [](const at::Tensor& self) -> at::Tensor {
        return run_or_cast(odnn_hijack::sigmoid_onednn, self);
    });
    m.impl("silu", [](const at::Tensor& self) -> at::Tensor {
        return run_or_cast(odnn_hijack::silu_onednn, self);
    });
    m.impl("leaky_relu", [](const at::Tensor& self, const at::Scalar& ns) -> at::Tensor {
        return run_or_cast(odnn_hijack::leaky_relu_onednn, self, ns);
    });
    m.impl("elu", [](const at::Tensor& self, const at::Scalar& a, const at::Scalar& s, const at::Scalar& is) -> at::Tensor {
        return run_or_cast(odnn_hijack::elu_onednn, self, a, s, is);
    });
    m.impl("hardswish", [](const at::Tensor& self) -> at::Tensor {
        return run_or_cast(odnn_hijack::hardswish_onednn, self);
    });
    m.impl("relu", [](const at::Tensor& self) -> at::Tensor {
        return run_or_cast(odnn_hijack::relu_onednn, self);
    });
    m.impl("clamp", [](const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max) -> at::Tensor {
        auto out = odnn_hijack::clamp_onednn(self, min, max);
        if (out.defined()) return out;
        auto s32 = self.to(at::kFloat);
        out = odnn_hijack::clamp_onednn(s32, min, max);
        return out.to(self.scalar_type());
    });
    m.impl("add.Tensor", [](const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) -> at::Tensor {
        auto out = odnn_hijack::add_onednn(a, b, alpha);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        out = odnn_hijack::add_onednn(a32, b32, alpha);
        return out.to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("sub.Tensor", [](const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) -> at::Tensor {
        auto out = odnn_hijack::sub_onednn(a, b, alpha);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        out = odnn_hijack::sub_onednn(a32, b32, alpha);
        return out.to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("mul.Tensor", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::mul_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        out = odnn_hijack::mul_onednn(a32, b32);
        return out.to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("div.Tensor", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::div_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        out = odnn_hijack::div_onednn(a32, b32);
        return out.to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("maximum", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::maximum_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        out = odnn_hijack::maximum_onednn(a32, b32);
        return out.to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("minimum", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::minimum_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        out = odnn_hijack::minimum_onednn(a32, b32);
        return out.to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("softmax.int", [](const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
        return run_or_cast(odnn_hijack::softmax_onednn, self, dim, dtype);
    });
    m.impl("log_softmax.int", [](const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
        return run_or_cast(odnn_hijack::log_softmax_onednn, self, dim, dtype);
    });
    m.impl("mm", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::matmul_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        return odnn_hijack::matmul_onednn(a32, b32).to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("bmm", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::matmul_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        return odnn_hijack::matmul_onednn(a32, b32).to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
    m.impl("matmul", [](const at::Tensor& a, const at::Tensor& b) -> at::Tensor {
        auto out = odnn_hijack::matmul_onednn(a, b);
        if (out.defined()) return out;
        auto a32 = a.to(at::kFloat), b32 = b.to(at::kFloat);
        return odnn_hijack::matmul_onednn(a32, b32).to(at::promote_types(a.scalar_type(), b.scalar_type()));
    });
}
