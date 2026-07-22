"""Lazy graph capture with JIT, Dynamo, and eager fallback tiers."""

import functools
import threading
import warnings

import torch
from torch.jit._trace import TracerWarning


class _RunMethod:
    """Graph execution methods used by :class:`GraphCaptureLite`."""

    JIT = 1
    DYNAMO = 2
    EAGER = 3

    @classmethod
    def label(cls, method):
        return {
            None: "pending",
            cls.JIT: "jit",
            cls.DYNAMO: "dynamo",
            cls.EAGER: "eager",
        }.get(method, "unknown")


class GraphCaptureLite:
    """Capture a model lazily on its first forward call.

    Capture first attempts ``torch.jit.trace`` and freeze, then Dynamo with a
    JIT backend, and finally falls back to the original eager ``forward``.
    """

    def __init__(self, model: torch.nn.Module, dtype=None):
        self.model = model
        self.dtype = dtype
        self.method = None
        self.lock = threading.Lock()
        self._original_forward = None
        self._captured_callable = None

    @staticmethod
    def _jit_compile(graph_module, example_inputs):
        """Compile a Dynamo FX graph with JIT when possible."""
        try:
            with torch.no_grad():
                traced = torch.jit.trace(graph_module.eval(), example_inputs)
                return torch.jit.freeze(traced)
        except Exception:
            return graph_module

    def _try_capture(self, *inputs, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=TracerWarning)
                traced = torch.jit.trace(self.model.eval(), inputs).eval()
                traced = torch.jit.freeze(traced)
            output = traced(*inputs, **kwargs)
            self._captured_callable = traced
            self._set_method(_RunMethod.JIT)
            return output
        except Exception:
            pass

        try:
            torch._dynamo.reset()
            dynamo_forward = torch._dynamo.optimize(
                self._jit_compile, dynamic=True
            )(self._original_forward)
            output = dynamo_forward(*inputs, **kwargs)
            self._captured_callable = dynamo_forward
            self._set_method(_RunMethod.DYNAMO)
            return output
        except Exception:
            torch._dynamo.reset()

        self._captured_callable = self._original_forward
        self._set_method(_RunMethod.EAGER)
        return self._captured_callable(*inputs, **kwargs)

    def _set_method(self, method):
        self.method = method
        report = getattr(self.model, "optimize_report", None)
        if report is not None:
            report.capture_method = _RunMethod.label(method)

    def __call__(self, original_forward):
        """Wrap ``original_forward`` with thread-safe lazy capture."""
        self._original_forward = original_forward

        @functools.wraps(original_forward)
        def captured_forward(*inputs, **kwargs):
            if torch.jit.is_tracing():
                return original_forward(*inputs, **kwargs)

            with torch.amp.autocast(
                "cpu",
                enabled=self.dtype in (torch.bfloat16, torch.float16),
                dtype=self.dtype,
            ):
                if self.method is not None:
                    return self._captured_callable(*inputs, **kwargs)

                with self.lock:
                    if self.method is not None:
                        return self._captured_callable(*inputs, **kwargs)
                    return self._try_capture(*inputs, **kwargs)

        return captured_forward
