#include "../utils.h"

namespace odnn_hijack {

// ---------------------------------------------------------------------------
// 通用 matmul（mm / bmm / matmul）
// 简化：只有 2D (mm) 和 3D (bmm) 直接走 oneDNN，其他回退 ATen
// ---------------------------------------------------------------------------
at::Tensor matmul_onednn(
        const at::Tensor& self,
        const at::Tensor& other) {
    if (!is_dnnl_friendly(self) || !is_dnnl_friendly(other)) return {};

    auto a = self.contiguous();
    auto b = other.contiguous();

    auto ndim_a = a.dim();
    auto ndim_b = b.dim();
    if (ndim_a < 2 || ndim_b < 2) return {};

    // 只处理 2D (mm) 和 3D (bmm) 的简单场景
    if (ndim_a > 3 || ndim_b > 3) return {};

    std::vector<int64_t> out_sizes;
    if (ndim_a == 2 && ndim_b == 2) {
        // mm: [m,k] @ [k,n] -> [m,n]
        out_sizes = {a.size(0), b.size(1)};
    } else if (ndim_a == 3 && ndim_b == 3) {
        // bmm: [b,m,k] @ [b,k,n] -> [b,m,n]
        if (a.size(0) != b.size(0) && a.size(0) != 1 && b.size(0) != 1) {
            return {}; // 不支持广播 batch
        }
        int64_t batch = std::max(a.size(0), b.size(0));
        out_sizes = {batch, a.size(ndim_a - 2), b.size(ndim_b - 1)};
    } else if (ndim_a == 2 && ndim_b == 3) {
        // [m,k] @ [b,k,n] -> [b,m,n]
        out_sizes = {b.size(0), a.size(0), b.size(2)};
    } else if (ndim_a == 3 && ndim_b == 2) {
        // [b,m,k] @ [k,n] -> [b,m,n]
        out_sizes = {a.size(0), a.size(1), b.size(1)};
    } else {
        return {}; // 更复杂的回退
    }

    auto dims_a = to_dims(a.sizes());
    auto dims_b = to_dims(b.sizes());
    auto out_dims = to_dims(out_sizes);

    auto src_mem = tensor_to_memory(a, dims_a);
    auto wei_mem = tensor_to_memory(b, dims_b);

    auto dst = at::empty(
        at::IntArrayRef(out_sizes.data(),
                        static_cast<int64_t>(out_sizes.size())),
        a.options());
    auto dst_mem = tensor_to_memory(dst, out_dims);

    auto pd = dnnl::matmul::primitive_desc(
        cpu_engine(),
        src_mem.get_desc(),
        wei_mem.get_desc(),
        dst_mem.get_desc());

    dnnl::matmul(pd).execute(
        dnnl::stream(cpu_engine()),
        {{DNNL_ARG_SRC, src_mem},
         {DNNL_ARG_WEIGHTS, wei_mem},
         {DNNL_ARG_DST, dst_mem}});

    return dst;
}

} // namespace odnn_hijack
