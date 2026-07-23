#include "utils.h"

// 声明各 kernel 模块的函数
namespace odnn_hijack {

// eltwise.cpp
at::Tensor gelu_onednn(const at::Tensor& input, c10::string_view approximate);
at::Tensor tanh_onednn(const at::Tensor& input);
at::Tensor sigmoid_onednn(const at::Tensor& input);
at::Tensor silu_onednn(const at::Tensor& input);
at::Tensor leaky_relu_onednn(const at::Tensor& input, const at::Scalar& negative_slope);
at::Tensor elu_onednn(const at::Tensor& input, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale);
at::Tensor hardswish_onednn(const at::Tensor& input);
at::Tensor relu_onednn(const at::Tensor& input);
at::Tensor abs_onednn(const at::Tensor& input);
at::Tensor exp_onednn(const at::Tensor& input);
at::Tensor log_onednn(const at::Tensor& input);
at::Tensor sqrt_onednn(const at::Tensor& input);

// pooling.cpp
at::Tensor max_pool2d_onednn(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
at::Tensor avg_pool2d_onednn(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override);
at::Tensor adaptive_avg_pool2d_onednn(const at::Tensor& input, at::IntArrayRef output_size);

// softmax.cpp
at::Tensor softmax_onednn(const at::Tensor& input, int64_t dim, c10::optional<at::ScalarType> dtype);
at::Tensor log_softmax_onednn(const at::Tensor& input, int64_t dim, c10::optional<at::ScalarType> dtype);

// binary.cpp
at::Tensor add_onednn(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor sub_onednn(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor mul_onednn(const at::Tensor& self, const at::Tensor& other);
at::Tensor div_onednn(const at::Tensor& self, const at::Tensor& other);
at::Tensor maximum_onednn(const at::Tensor& self, const at::Tensor& other);
at::Tensor minimum_onednn(const at::Tensor& self, const at::Tensor& other);

// unary.cpp
at::Tensor clamp_onednn(const at::Tensor& input, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max);

// normalization.cpp
at::Tensor layer_norm_onednn(const at::Tensor& input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, double eps);

// matmul.cpp
at::Tensor matmul_onednn(const at::Tensor& self, const at::Tensor& other);

} // namespace odnn_hijack

// ---------------------------------------------------------------------------
// 辅助宏：oneDNN kernel 失败时回退到 ATen 原生实现
// ---------------------------------------------------------------------------
#define ODNN_HIJACK_OR_FALLBACK(op_name, fn, ...)           \
    at::Tensor result = odnn_hijack::fn(__VA_ARGS__);       \
    if (result.defined()) return result;                    \
    return at::op_name(__VA_ARGS__);

// ---------------------------------------------------------------------------
// 注册所有 aten 算子劫持
// ---------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, CPU, m) {

    // === 激活函数 ===
    // gelu(Tensor self, *, str approximate='none') -> Tensor
    m.impl("gelu", [](const at::Tensor& self, c10::string_view approximate) -> at::Tensor {
        auto out = odnn_hijack::gelu_onednn(self, approximate);
        return out.defined() ? out : at::gelu(self, std::string(approximate));
    });

    // tanh(Tensor self) -> Tensor
    m.impl("tanh", [](const at::Tensor& self) -> at::Tensor {
        auto out = odnn_hijack::tanh_onednn(self);
        return out.defined() ? out : at::tanh(self);
    });

    // sigmoid(Tensor self) -> Tensor
    m.impl("sigmoid", [](const at::Tensor& self) -> at::Tensor {
        auto out = odnn_hijack::sigmoid_onednn(self);
        return out.defined() ? out : at::sigmoid(self);
    });

    // silu(Tensor self) -> Tensor
    m.impl("silu", [](const at::Tensor& self) -> at::Tensor {
        auto out = odnn_hijack::silu_onednn(self);
        return out.defined() ? out : at::silu(self);
    });

    // leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
    m.impl("leaky_relu", [](const at::Tensor& self, const at::Scalar& negative_slope) -> at::Tensor {
        auto out = odnn_hijack::leaky_relu_onednn(self, negative_slope);
        return out.defined() ? out : at::leaky_relu(self, negative_slope);
    });

    // elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    m.impl("elu", [](const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) -> at::Tensor {
        auto out = odnn_hijack::elu_onednn(self, alpha, scale, input_scale);
        return out.defined() ? out : at::elu(self, alpha, scale, input_scale);
    });

    // hardswish(Tensor self) -> Tensor
    m.impl("hardswish", [](const at::Tensor& self) -> at::Tensor {
        auto out = odnn_hijack::hardswish_onednn(self);
        return out.defined() ? out : at::hardswish(self);
    });

    // relu(Tensor self) -> Tensor
    m.impl("relu", [](const at::Tensor& self) -> at::Tensor {
        auto out = odnn_hijack::relu_onednn(self);
        return out.defined() ? out : at::relu(self);
    });

    // === 一元逐元素（注意：abs/exp/log/sqrt 有复杂 typed 重载，暂不劫持） ===
    // clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    m.impl("clamp", [](const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max) -> at::Tensor {
        auto out = odnn_hijack::clamp_onednn(self, min, max);
        return out.defined() ? out : at::clamp(self, min, max);
    });

    // === 二元逐元素 ===
    // add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    m.impl("add.Tensor", [](const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) -> at::Tensor {
        auto out = odnn_hijack::add_onednn(self, other, alpha);
        return out.defined() ? out : at::add(self, other, alpha);
    });

    // sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
    m.impl("sub.Tensor", [](const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) -> at::Tensor {
        auto out = odnn_hijack::sub_onednn(self, other, alpha);
        return out.defined() ? out : at::sub(self, other, alpha);
    });

    // mul.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("mul.Tensor", [](const at::Tensor& self, const at::Tensor& other) -> at::Tensor {
        auto out = odnn_hijack::mul_onednn(self, other);
        return out.defined() ? out : at::mul(self, other);
    });

    // div.Tensor(Tensor self, Tensor other) -> Tensor
    m.impl("div.Tensor", [](const at::Tensor& self, const at::Tensor& other) -> at::Tensor {
        auto out = odnn_hijack::div_onednn(self, other);
        return out.defined() ? out : at::div(self, other);
    });

    // maximum(Tensor self, Tensor other) -> Tensor
    m.impl("maximum", [](const at::Tensor& self, const at::Tensor& other) -> at::Tensor {
        auto out = odnn_hijack::maximum_onednn(self, other);
        return out.defined() ? out : at::maximum(self, other);
    });

    // minimum(Tensor self, Tensor other) -> Tensor
    m.impl("minimum", [](const at::Tensor& self, const at::Tensor& other) -> at::Tensor {
        auto out = odnn_hijack::minimum_onednn(self, other);
        return out.defined() ? out : at::minimum(self, other);
    });

    // === 池化 ===
    // === 池化（PyTorch 已有 oneDNN 优化，暂不劫持） ===

    // === softmax ===
    // _softmax(Tensor self, int dim, bool half_to_float) -> Tensor  (older)
    // softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor  (newer)
    m.impl("softmax.int", [](const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
        auto out = odnn_hijack::softmax_onednn(self, dim, dtype);
        return out.defined() ? out : at::softmax(self, dim, dtype);
    });

    // log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
    m.impl("log_softmax.int", [](const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
        auto out = odnn_hijack::log_softmax_onednn(self, dim, dtype);
        return out.defined() ? out : at::log_softmax(self, dim, dtype);
    });

    // === 归一化 ===
    // native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
    m.impl("native_layer_norm", [](const at::Tensor& input, c10::SymIntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& bias, double eps) -> std::tuple<at::Tensor, at::Tensor, at::Tensor> {
        std::vector<int64_t> ns;
        ns.reserve(normalized_shape.size());
        for (const auto& s : normalized_shape) {
            ns.push_back(s.expect_int());
        }
        auto out = odnn_hijack::layer_norm_onednn(input, ns, weight, bias, eps);
        if (out.defined()) {
            // 返回 (output, mean, rstd) — mean 和 rstd 给空 tensor
            auto dummy = at::empty({0}, input.options());
            return {out, dummy, dummy};
        }
        return at::native_layer_norm_symint(input, normalized_shape, weight, bias, eps);
    });

    // === 矩阵运算 ===
    // mm(Tensor self, Tensor mat2) -> Tensor
    m.impl("mm", [](const at::Tensor& self, const at::Tensor& mat2) -> at::Tensor {
        auto out = odnn_hijack::matmul_onednn(self, mat2);
        return out.defined() ? out : at::mm(self, mat2);
    });

    // bmm(Tensor self, Tensor mat2) -> Tensor
    m.impl("bmm", [](const at::Tensor& self, const at::Tensor& mat2) -> at::Tensor {
        auto out = odnn_hijack::matmul_onednn(self, mat2);
        return out.defined() ? out : at::bmm(self, mat2);
    });

    // matmul(Tensor self, Tensor other) -> Tensor
    m.impl("matmul", [](const at::Tensor& self, const at::Tensor& other) -> at::Tensor {
        auto out = odnn_hijack::matmul_onednn(self, other);
        return out.defined() ? out : at::matmul(self, other);
    });

}
