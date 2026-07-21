import torch
from torch import nn

from . import prepack_backend


class ODNNConvolutionOpContext(nn.Module):
    """Inference context holding a Conv2d weight packed for oneDNN."""

    def __init__(
        self,
        packed_module,
        *,
        channels_last,
        cpp_context,
        input_size,
        module,
    ):
        super().__init__()
        self.packed_module = packed_module
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
        cpp_op_context=False,
        input_size=(),
    ):
        packed_module = prepack_backend.prepack(module)
        cpp_context = None
        if cpp_op_context:
            from .cpp_extension import load_cpp_extension

            load_cpp_extension()
            cpp_context = torch.classes.odnn_prepack.ConvolutionOpContext(
                packed_module.weight,
                packed_module.bias,
                list(module.stride),
                list(module.padding),
                list(module.dilation),
                module.groups,
                list(input_size),
            )
        return cls(
            packed_module,
            channels_last=channels_last,
            cpp_context=cpp_context,
            input_size=input_size,
            module=module,
        )

    def run(self, input, *, keep_layout=False):
        if self.channels_last and input.dim() == 4 and not input.is_mkldnn:
            if not input.is_contiguous(memory_format=torch.channels_last):
                input = input.contiguous(memory_format=torch.channels_last)
        if self.cpp_context is not None:
            output = self.cpp_context.run(
                input if input.is_mkldnn else input.to_mkldnn()
            )
            return output if keep_layout else prepack_backend.to_dense(output)
        if keep_layout:
            return prepack_backend.run_mkldnn(self.packed_module, input)
        return prepack_backend.run(self.packed_module, input)

    def get_packed_weight(self):
        return self.packed_module.weight


class ODNNLinearOpContext(nn.Module):
    """Inference context holding a Linear weight packed for oneDNN."""

    def __init__(self, packed_module, *, cpp_context, input_size, module):
        super().__init__()
        self.packed_module = packed_module
        self.cpp_context = cpp_context
        self.input_size = tuple(input_size)
        self.in_features = module.in_features
        self.out_features = module.out_features

    @classmethod
    def create_context(cls, module, *, cpp_op_context=False, input_size=()):
        packed_module = prepack_backend.prepack(module)
        cpp_context = None
        if cpp_op_context:
            from .cpp_extension import load_cpp_extension

            load_cpp_extension()
            cpp_context = torch.classes.odnn_prepack.LinearOpContext(
                packed_module.weight,
                packed_module.bias,
                list(input_size),
            )
        return cls(
            packed_module,
            cpp_context=cpp_context,
            input_size=input_size,
            module=module,
        )

    def run(self, input):
        if self.cpp_context is not None:
            output = self.cpp_context.run(
                input if input.is_mkldnn else input.to_mkldnn()
            )
            return prepack_backend.to_dense(output)
        return prepack_backend.run(self.packed_module, input)

    def get_packed_weight(self):
        return self.packed_module.weight
