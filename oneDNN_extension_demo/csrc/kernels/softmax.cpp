#include "../utils.h"

namespace odnn_hijack {

// ---------------------------------------------------------------------------
// softmax / log_softmax
// ---------------------------------------------------------------------------
at::Tensor softmax_onednn(
        const at::Tensor& input,
        int64_t dim,
        c10::optional<at::ScalarType> dtype) {
    if (!is_dnnl_friendly(input)) return {};

    // 处理负数维度和 dtype
    auto ndim = input.dim();
    if (dim < 0) dim += ndim;

    // oneDNN softmax 要求在最后一个维度（axis = ndims - 1）或 channels (1)
    // 如果 dim 不是最后一个维度，需要 transpose
    // 为简化，先只处理 dim == ndim-1 或 dim == 1 的情况
    if (dim != ndim - 1 && dim != 1 && ndim > 2) return {};

    auto src = input;
    bool need_cast = dtype.has_value() && dtype.value() != input.scalar_type();
    if (need_cast) src = src.to(dtype.value());

    // oneDNN softmax axis: 0-based
    int64_t axis = dim;

    auto dims = to_dims(src.sizes());
    auto src_mem = tensor_to_memory(src, dims);
    auto dst = at::empty_like(src);
    auto dst_mem = tensor_to_memory(dst, dims);

    auto pd = dnnl::softmax_forward::primitive_desc(
        cpu_engine(),
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::softmax_accurate,
        src_mem.get_desc(),
        dst_mem.get_desc(),
        static_cast<int>(axis));

    dnnl::softmax_forward(pd).execute(
        dnnl::stream(cpu_engine()),
        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    if (need_cast) return dst.to(input.scalar_type());
    return dst;
}

at::Tensor log_softmax_onednn(
        const at::Tensor& input,
        int64_t dim,
        c10::optional<at::ScalarType> dtype) {
    if (!is_dnnl_friendly(input)) return {};

    auto ndim = input.dim();
    if (dim < 0) dim += ndim;
    if (dim != ndim - 1 && dim != 1 && ndim > 2) return {};

    auto src = input;
    bool need_cast = dtype.has_value() && dtype.value() != input.scalar_type();
    if (need_cast) src = src.to(dtype.value());

    int64_t axis = dim;

    auto dims = to_dims(src.sizes());
    auto src_mem = tensor_to_memory(src, dims);
    auto dst = at::empty_like(src);
    auto dst_mem = tensor_to_memory(dst, dims);

    auto pd = dnnl::softmax_forward::primitive_desc(
        cpu_engine(),
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::softmax_log,
        src_mem.get_desc(),
        dst_mem.get_desc(),
        static_cast<int>(axis));

    dnnl::softmax_forward(pd).execute(
        dnnl::stream(cpu_engine()),
        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    if (need_cast) return dst.to(input.scalar_type());
    return dst;
}

} // namespace odnn_hijack
