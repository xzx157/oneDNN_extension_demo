"""oneDNN extension demo — import 时自动注册算子并劫持 aten kernel。"""

from . import ops              # 注册 odnn::* 自定义算子

# 加载 C++ aten 劫持扩展（dnnl::* 直接调用替代 aten::*）
from .cpp_extension import load_hijack_extension

try:
    load_hijack_extension()
except Exception as e:
    import warnings
    warnings.warn(
        f"oneDNN aten hijack extension could not be loaded: {e}. "
        "The package will work but without operator-level hijack optimization.",
        RuntimeWarning,
        stacklevel=2,
    )

from .frontend import optimize

__all__ = ["optimize"]
