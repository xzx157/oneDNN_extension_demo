import torch
from torch import nn


class ODNNConvolutionOpContext(nn.Module):
    """Inference context holding a Conv2d execution backend and weight."""

    def __init__(
        self,
        *,
        channels_last,
        cpp_context,
        input_size,
        module,
    ):
        super().__init__()
        self.cpp_context = cpp_context
        self.channels_last = channels_last
        self.input_size = tuple(input_size)
        self.weight_size = tuple(module.weight.shape)
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups

    @classmethod
    def create_context(
        cls,
        module,
        *,
        channels_last=False,
        input_size=(),
    ):
        from .cpp_extension import load_cpp_extension

        load_cpp_extension()
        weight = module.weight.detach()
        weight = weight.contiguous(
            memory_format=(
                torch.channels_last
                if channels_last
                else torch.contiguous_format
            )
        )
        cpp_context = torch.classes.odnn_prepack.ConvolutionOpContext(
            weight,
            module.bias,
            list(module.stride),
            list(module.padding),
            list(module.dilation),
            module.groups,
            list(input_size),
            channels_last,
        )
        return cls(
            channels_last=channels_last,
            cpp_context=cpp_context,
            input_size=input_size,
            module=module,
        )

    def run(self, input):
        if self.channels_last and input.dim() == 4 and not input.is_mkldnn:
            if not input.is_contiguous(memory_format=torch.channels_last):
                input = input.contiguous(memory_format=torch.channels_last)
        if input.is_mkldnn:
            input = input.to_dense()
        return self.cpp_context.run(input)

    def get_packed_weight(self):
        return self.cpp_context.get_packed_weight()

    def uses_native_dnnl(self):
        return bool(self.cpp_context.uses_native_dnnl())

    def packed_weight_bytes(self):
        return int(self.cpp_context.packed_weight_bytes())

    def runtime_stats(self):
        return {
            "native_runs": int(self.cpp_context.native_runs()),
            "fallback_runs": int(self.cpp_context.fallback_runs()),
        }


class ODNNLinearOpContext(nn.Module):
    """Inference context holding a Linear weight packed for oneDNN."""

    def __init__(self, *, cpp_context, input_size, module):
        super().__init__()
        self.cpp_context = cpp_context
        self.input_size = tuple(input_size)
        self.in_features = module.in_features
        self.out_features = module.out_features

    @classmethod
    def create_context(cls, module, *, input_size=()):
        from .cpp_extension import load_cpp_extension

        load_cpp_extension()
        cpp_context = torch.classes.odnn_prepack.LinearOpContext(
            module.weight.detach().contiguous(),
            module.bias,
            list(input_size),
        )
        return cls(
            cpp_context=cpp_context,
            input_size=input_size,
            module=module,
        )

    def run(self, input):
        if input.is_mkldnn:
            input = input.to_dense()
        return self.cpp_context.run(input)

    def get_packed_weight(self):
        return self.cpp_context.get_packed_weight()
