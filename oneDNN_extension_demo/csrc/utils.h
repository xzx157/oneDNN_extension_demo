#pragma once

#include <dnnl.hpp>
#include <torch/library.h>
#include <ATen/Functions.h>

#include <unordered_map>
#include <vector>

namespace odnn_hijack {

// ---------------------------------------------------------------------------
// 公共工具
// ---------------------------------------------------------------------------

/// 全局 oneDNN engine（单例，线程安全）
inline dnnl::engine& cpu_engine() {
    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

/// at::IntArrayRef → dnnl::memory::dims
inline dnnl::memory::dims to_dims(at::IntArrayRef sizes) {
    return dnnl::memory::dims(sizes.begin(), sizes.end());
}

/// 将 at::Tensor 包装为 dnnl::memory（不拷贝数据）
inline dnnl::memory tensor_to_memory(
        const at::Tensor& t,
        const dnnl::memory::dims& dims,
        const dnnl::memory::format_tag& tag = dnnl::memory::format_tag::undef) {
    auto dt = dnnl::memory::data_type::f32;
    switch (t.scalar_type()) {
        case at::kFloat:   dt = dnnl::memory::data_type::f32;  break;
        case at::kBFloat16: dt = dnnl::memory::data_type::bf16; break;
        case at::kHalf:    dt = dnnl::memory::data_type::f16;  break;
        case at::kInt:     dt = dnnl::memory::data_type::s32;  break;
        case at::kByte:    dt = dnnl::memory::data_type::u8;   break;
        default:
            TORCH_CHECK(false, "unsupported dtype for oneDNN: ", t.scalar_type());
    }
    auto desc = tag == dnnl::memory::format_tag::undef
        ? dnnl::memory::desc(dims, dt, to_dims(t.strides()))
        : dnnl::memory::desc(dims, dt, tag);
    return dnnl::memory(desc, cpu_engine(), t.data_ptr());
}

/// 检查 tensor 是否可用 oneDNN 处理
inline bool is_dnnl_friendly(const at::Tensor& t) {
    return t.defined() && t.device().is_cpu() &&
           t.layout() == c10::kStrided && t.numel() > 0;
}

/// 检查是否所有 tensor 都适合 oneDNN
template <typename... Tensors>
inline bool all_dnnl_friendly(const at::Tensor& first, const Tensors&... rest) {
    if (!is_dnnl_friendly(first)) return false;
    if constexpr (sizeof...(rest) > 0) return all_dnnl_friendly(rest...);
    return true;
}

} // namespace odnn_hijack
