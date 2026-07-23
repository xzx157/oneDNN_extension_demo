#include "../utils.h"
#include <cstdio>

namespace odnn_hijack {

// ---------------------------------------------------------------------------
// 通用 binary 操作
// ---------------------------------------------------------------------------
static int g_binary_calls = 0;
at::Tensor binary_forward(
        const at::Tensor& a,
        const at::Tensor& b,
        dnnl::algorithm algo) {
    if (!is_dnnl_friendly(a) || !is_dnnl_friendly(b)) return {};

    g_binary_calls++;
    if (g_binary_calls == 1) {
        fprintf(stderr, "[ODNN_HIJACK] binary_forward() CALLED — C++ dnnl::binary is ACTIVE\n");
    }

    auto dims_a = to_dims(a.sizes());
    auto dims_b = to_dims(b.sizes());

    // oneDNN binary 要求两个输入维度相同或可广播
    // 先做广播对齐
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    auto src0_mem = tensor_to_memory(a_contig, to_dims(a_contig.sizes()));
    auto src1_mem = tensor_to_memory(b_contig, to_dims(b_contig.sizes()));

    // 输出形状：广播后的
    auto dst = at::empty_like(a_contig);
    // 如果 b 比 a 大，输出应该和 b 一样
    if (b_contig.numel() > a_contig.numel()) {
        dst = at::empty_like(b_contig);
    }

    auto dst_mem = tensor_to_memory(dst, to_dims(dst.sizes()));

    auto pd = dnnl::binary::primitive_desc(
        cpu_engine(),
        algo,
        src0_mem.get_desc(),
        src1_mem.get_desc(),
        dst_mem.get_desc());

    dnnl::binary(pd).execute(
        dnnl::stream(cpu_engine()),
        {{DNNL_ARG_SRC_0, src0_mem},
         {DNNL_ARG_SRC_1, src1_mem},
         {DNNL_ARG_DST, dst_mem}});

    // 如果广播改变了形状，恢复到期望的输出形状
    if (dst.sizes() != a.sizes() && a.numel() > 0) {
        return dst.broadcast_to(a.sizes()).contiguous();
    }
    return dst;
}

// ---------------------------------------------------------------------------
// add / sub / mul / div / maximum / minimum
// ---------------------------------------------------------------------------

at::Tensor add_onednn(
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha) {
    float a = alpha.to<float>();
    if (a != 1.0f) {
        // oneDNN binary 不支持 alpha 参数，需手动乘
        auto scaled = other.mul(a);
        return binary_forward(self, scaled, dnnl::algorithm::binary_add);
    }
    return binary_forward(self, other, dnnl::algorithm::binary_add);
}

at::Tensor sub_onednn(
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha) {
    float a = alpha.to<float>();
    if (a != 1.0f) {
        auto scaled = other.mul(a);
        return binary_forward(self, scaled, dnnl::algorithm::binary_sub);
    }
    return binary_forward(self, other, dnnl::algorithm::binary_sub);
}

at::Tensor mul_onednn(
        const at::Tensor& self,
        const at::Tensor& other) {
    return binary_forward(self, other, dnnl::algorithm::binary_mul);
}

at::Tensor div_onednn(
        const at::Tensor& self,
        const at::Tensor& other) {
    return binary_forward(self, other, dnnl::algorithm::binary_div);
}

at::Tensor maximum_onednn(
        const at::Tensor& self,
        const at::Tensor& other) {
    return binary_forward(self, other, dnnl::algorithm::binary_max);
}

at::Tensor minimum_onednn(
        const at::Tensor& self,
        const at::Tensor& other) {
    return binary_forward(self, other, dnnl::algorithm::binary_min);
}

} // namespace odnn_hijack
