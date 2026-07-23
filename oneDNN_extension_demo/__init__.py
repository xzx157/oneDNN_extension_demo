"""Public entry points for the oneDNN extension demo."""

from . import ops
from .frontend import optimize

# 导入时自动启用 C++ aten hijack（dnnl::* 直接调用）
from .cpp_extension import load_hijack_extension
load_hijack_extension()

__all__ = ["optimize", "ops"]


__all__ = ["enable_aten_hijack", "optimize", "ops"]
