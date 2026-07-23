#include "../utils.h"
#include <cstdio>

namespace odnn_hijack {

// 调试计数器：证明 C++ kernel 被真实调用
int g_eltwise_calls = 0;

// ---------------------------------------------------------------------------
// 通用 eltwise 执行模板
// ---------------------------------------------------------------------------
at::Tensor eltwise_forward(
        const at::Tensor& input,
        dnnl::algorithm algo,
        float alpha = 0.f,
        float beta = 0.f) {
    if (!is_dnnl_friendly(input)) {
        return {};
    }

    g_eltwise_calls++;
    if (g_eltwise_calls == 1) {
        fprintf(stderr, "[ODNN_HIJACK] eltwise_forward() CALLED — C++ dnnl::eltwise is ACTIVE\n");
    }

    auto dims = to_dims(input.sizes());
    auto src_mem = tensor_to_memory(input, dims);
    auto dst = at::empty_like(input);
    auto dst_mem = tensor_to_memory(dst, dims);

    auto pd = dnnl::eltwise_forward::primitive_desc(
        cpu_engine(),
        dnnl::prop_kind::forward_inference,
        algo,
        src_mem.get_desc(),
        dst_mem.get_desc(),
        alpha,
        beta);

    dnnl::eltwise_forward(pd).execute(
        dnnl::stream(cpu_engine()),
        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    return dst;
}

// ---------------------------------------------------------------------------
// 各激活函数
// ---------------------------------------------------------------------------

at::Tensor gelu_onednn(const at::Tensor& input, c10::string_view approximate) {
    // oneDNN 支持 gelu_erf 和 gelu_tanh
    auto algo = (approximate == "tanh")
        ? dnnl::algorithm::eltwise_gelu_tanh
        : dnnl::algorithm::eltwise_gelu_erf;
    return eltwise_forward(input, algo);
}

at::Tensor tanh_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_tanh);
}

at::Tensor sigmoid_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_logistic);
}

at::Tensor silu_onednn(const at::Tensor& input) {
    // oneDNN eltwise_swish = silu (x * sigmoid(x))
    return eltwise_forward(input, dnnl::algorithm::eltwise_swish);
}

at::Tensor leaky_relu_onednn(
        const at::Tensor& input,
        const at::Scalar& negative_slope) {
    // oneDNN leaky_relu 用 eltwise_relu + alpha
    float alpha = negative_slope.to<float>();
    return eltwise_forward(input, dnnl::algorithm::eltwise_relu, alpha);
}

at::Tensor elu_onednn(
        const at::Tensor& input,
        const at::Scalar& alpha,
        const at::Scalar& scale,
        const at::Scalar& input_scale) {
    float a = alpha.to<float>();
    return eltwise_forward(input, dnnl::algorithm::eltwise_elu, a);
}

at::Tensor hardswish_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_hardswish);
}

at::Tensor relu_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_relu);
}

at::Tensor abs_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_abs);
}

at::Tensor exp_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_exp);
}

at::Tensor log_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_log);
}

at::Tensor sqrt_onednn(const at::Tensor& input) {
    return eltwise_forward(input, dnnl::algorithm::eltwise_sqrt);
}

} // namespace odnn_hijack

// 导出计数器供外部查询
int odnn_hijack_get_eltwise_call_count() {
    return odnn_hijack::g_eltwise_calls;
}
