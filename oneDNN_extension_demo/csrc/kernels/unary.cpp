#include "../utils.h"

namespace odnn_hijack {

// ---------------------------------------------------------------------------
// clamp (clip) — 用 eltwise 实现
// ---------------------------------------------------------------------------
at::Tensor clamp_onednn(
        const at::Tensor& input,
        const c10::optional<at::Scalar>& min,
        const c10::optional<at::Scalar>& max) {
    if (!is_dnnl_friendly(input)) return {};

    float alpha = min.has_value() ? min->to<float>() : -INFINITY;
    float beta  = max.has_value() ? max->to<float>() : INFINITY;

    // oneDNN clip: 对于无穷边界用 clip_v2 或 clip
    // clip_v2 不支持 INFINITY，所以用条件判断
    if (!min.has_value() && !max.has_value()) return input.clone();

    auto dims = to_dims(input.sizes());
    auto src_mem = tensor_to_memory(input, dims);
    auto dst = at::empty_like(input);
    auto dst_mem = tensor_to_memory(dst, dims);

    try {
        auto pd = dnnl::eltwise_forward::primitive_desc(
            cpu_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_clip_v2,
            src_mem.get_desc(),
            dst_mem.get_desc(),
            alpha,
            beta);

        dnnl::eltwise_forward(pd).execute(
            dnnl::stream(cpu_engine()),
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    } catch (const dnnl::error&) {
        // clip_v2 在某些边界值下失败，回退
        return {};
    }

    return dst;
}

} // namespace odnn_hijack
