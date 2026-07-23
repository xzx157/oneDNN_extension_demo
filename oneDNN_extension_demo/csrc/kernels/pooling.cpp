#include "../utils.h"

namespace odnn_hijack {

// 池化算子目前由 ATen oneDNN 原生实现处理（已有优化），暂不劫持

at::Tensor max_pool2d_onednn(
        const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
        at::IntArrayRef, at::IntArrayRef, bool) { return {}; }

at::Tensor avg_pool2d_onednn(
        const at::Tensor&, at::IntArrayRef, at::IntArrayRef,
        at::IntArrayRef, bool, bool, c10::optional<int64_t>) { return {}; }

at::Tensor adaptive_avg_pool2d_onednn(
        const at::Tensor&, at::IntArrayRef) { return {}; }

} // namespace odnn_hijack
