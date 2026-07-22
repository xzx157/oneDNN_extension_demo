"""惰性图捕获：JIT trace → dynamo → eager 三级级联回退。"""

import functools
import threading
import warnings

import torch
from torch.jit._trace import TracerWarning


class _RunMethod:
    """记录当前图捕获使用了哪种方式。"""
    JIT = 1
    DYNAMO = 2
    EAGER = 3

    @classmethod
    def label(cls, method):
        return {
            cls.JIT: "jit",
            cls.DYNAMO: "dynamo",
            cls.EAGER: "eager",
        }.get(method, "unknown")


class GraphCaptureLite:
    """
    惰性图捕获包装器。

    不在 optimize() 时 trace，而在第一次 forward() 调用时 trace。
    用户不需要传 sample_input，接口更简洁。

    级联策略：
      Tier 1: torch.jit.trace + freeze → 性能最好
      Tier 2: torch._dynamo + JIT backend → 处理动态控制流
      Tier 3: eager fallback             → 兜底
    """

    def __init__(self, model: torch.nn.Module, dtype=None):
        self.model = model          # 模型引用（不 deepcopy）
        self.dtype = dtype          # 混合精度类型，None 表示不做
        self.method = None          # None=未捕获, JIT/DYNAMO/EAGER
        self.lock = threading.Lock()

    @staticmethod
    def _jit_compile(gm, example_inputs):
        """dynamo compiler backend：对 FX GraphModule 做 JIT trace + freeze。"""
        try:
            with torch.no_grad():
                traced = torch.jit.trace(gm.eval(), example_inputs)
                traced = torch.jit.freeze(traced)
            return traced
        except Exception:
            return gm

    def _try_capture(self, *input, **kwargs):
        """级联尝试 JIT → dynamo → eager。"""

        # ---- Tier 1: JIT trace + freeze ----
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=TracerWarning)
                traced = torch.jit.trace(self.model.eval(), input).eval()
                traced = torch.jit.freeze(traced)
            output = traced(*input, **kwargs)
            self.model = traced
            self.method = _RunMethod.JIT
            return output
        except Exception:
            pass

        # ---- Tier 2: torch._dynamo + JIT backend ----
        try:
            torch._dynamo.reset()
            dynamo_model = torch._dynamo.optimize(
                self._jit_compile, dynamic=True
            )(self.model)
            output = dynamo_model(*input, **kwargs)
            self.model = dynamo_model
            self.method = _RunMethod.DYNAMO
            return output
        except Exception:
            pass

        # ---- Tier 3: eager fallback ----
        torch._dynamo.reset()
        self.method = _RunMethod.EAGER
        return self.model(*input, **kwargs)

    def __call__(self, original_forward):
        """作为装饰器使用：model.forward = GraphCaptureLite(model)(model.forward)"""

        @functools.wraps(original_forward)
        def captured_forward(*input, **kwargs):
            # 如果外层已经在 trace 这个模型，不要再套一层
            if torch.jit.is_tracing():
                return original_forward(*input, **kwargs)

            with torch.amp.autocast('cpu',
                enabled=(self.dtype in (torch.bfloat16, torch.float16)),
                dtype=self.dtype,
            ):
                if self.method is not None:
                    return self.model(*input, **kwargs)

                # 双重检查锁：防止多线程重复捕获
                with self.lock:
                    if self.method is not None:
                        return self.model(*input, **kwargs)
                    return self._try_capture(*input, **kwargs)

        return captured_forward
