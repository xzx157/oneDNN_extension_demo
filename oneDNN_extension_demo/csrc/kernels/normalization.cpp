#include "../utils.h"

namespace odnn_hijack {

// ---------------------------------------------------------------------------
// layer_norm
// ---------------------------------------------------------------------------
at::Tensor layer_norm_onednn(
        const at::Tensor& input,
        at::IntArrayRef normalized_shape,
        const c10::optional<at::Tensor>& weight,
        const c10::optional<at::Tensor>& bias,
        double eps) {
    if (!is_dnnl_friendly(input)) return {};

    auto ndim = input.dim();
    int64_t norm_ndim = normalized_shape.size();
    int64_t axis = ndim - norm_ndim;

    // oneDNN layer_norm 要求 axis 从 1 开始
    if (axis < 1) return {};

    auto dims = to_dims(input.sizes());
    auto src_mem = tensor_to_memory(input.contiguous(), dims);

    // weight 和 bias 的形状
    auto w_dims = to_dims(normalized_shape);
    dnnl::memory w_mem, b_mem;
    bool has_w = weight.has_value() && weight->defined();
    bool has_b = bias.has_value() && bias->defined();

    // 确保 weight/bias 在 CPU 上
    at::Tensor w_cpu, b_cpu;
    if (has_w) {
        w_cpu = weight->contiguous().cpu();
        w_mem = tensor_to_memory(w_cpu, w_dims, dnnl::memory::format_tag::x);
    }
    if (has_b) {
        b_cpu = bias->contiguous().cpu();
        b_mem = tensor_to_memory(b_cpu, w_dims, dnnl::memory::format_tag::x);
    }

    auto dst = at::empty_like(input);
    auto dst_mem = tensor_to_memory(dst, dims);

    // mean 和 variance（不需要保存，但 oneDNN 要求提供）
    auto mean_dims = dnnl::memory::dims(dims.begin(), dims.begin() + axis);
    auto mean = at::empty(
        at::IntArrayRef(mean_dims.data(), mean_dims.size()),
        input.options());
    auto var = at::empty_like(mean);
    auto mean_mem = tensor_to_memory(mean, mean_dims, dnnl::memory::format_tag::x);
    auto var_mem  = tensor_to_memory(var, mean_dims, dnnl::memory::format_tag::x);

    auto pd = dnnl::layer_normalization_forward::primitive_desc(
        cpu_engine(),
        dnnl::prop_kind::forward_inference,
        src_mem.get_desc(),
        dst_mem.get_desc(),
        mean_mem.get_desc(),  // stat_desc (dims not normalized)
        static_cast<float>(eps),
        dnnl::normalization_flags::use_scale |
        dnnl::normalization_flags::use_shift);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_MEAN, mean_mem},
        {DNNL_ARG_VARIANCE, var_mem}};

    if (has_w) args[DNNL_ARG_SCALE] = w_mem;
    if (has_b) args[DNNL_ARG_SHIFT] = b_mem;

    dnnl::layer_normalization_forward(pd).execute(
        dnnl::stream(cpu_engine()), args);

    return dst;
}

} // namespace odnn_hijack
