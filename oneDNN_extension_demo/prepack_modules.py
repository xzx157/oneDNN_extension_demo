import torch
import torch.nn.functional as F
from torch import nn

from .prepack_context import ODNNConvolutionOpContext, ODNNLinearOpContext
from . import prepack_backend


class _ODNNPrepackedConv2d(nn.Module):
    def __init__(
        self,
        module,
        *,
        channels_last=False,
        cpp_op_context=False,
        preserve_mkldnn_layout=False,
    ):
        super().__init__()
        self.preserve_mkldnn_layout = preserve_mkldnn_layout
        self.ctx = ODNNConvolutionOpContext.create_context(
            module,
            channels_last=channels_last,
            cpp_op_context=cpp_op_context,
            input_size=getattr(module, "_odnn_sample_input_size", ()),
        )

    def forward(self, input):
        return self.ctx.run(
            input,
            keep_layout=self.preserve_mkldnn_layout,
        )


class _ODNNPrepackedLinear(nn.Module):
    def __init__(self, module, *, cpp_op_context=False):
        super().__init__()
        self.ctx = ODNNLinearOpContext.create_context(
            module,
            cpp_op_context=cpp_op_context,
            input_size=getattr(module, "_odnn_sample_input_size", ()),
        )

    def forward(self, input):
        return self.ctx.run(input)


class _ODNNMKLDNNBatchNorm2d(nn.Module):
    """BatchNorm boundary that preserves MKLDNN layout for the next Conv."""

    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias
        self.register_buffer("running_mean", module.running_mean)
        self.register_buffer("running_var", module.running_var)
        self.momentum = module.momentum
        self.eps = module.eps
        self.training = module.training
        self._mkldnn_supported = None

    def forward(self, input):
        if self._mkldnn_supported is not False:
            try:
                output = F.batch_norm(
                    input if input.is_mkldnn else input.to_mkldnn(),
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    self.training,
                    self.momentum,
                    self.eps,
                )
                self._mkldnn_supported = True
                return output if output.is_mkldnn else output.to_mkldnn()
            except (NotImplementedError, RuntimeError):
                self._mkldnn_supported = False

        output = F.batch_norm(
            prepack_backend.to_dense(input),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
        return output.to_mkldnn()


class _ODNNMKLDNNReLU(nn.Module):
    """ReLU boundary that preserves MKLDNN layout for the next Conv."""

    def __init__(self, module):
        super().__init__()
        self.inplace = module.inplace

    def forward(self, input):
        output = F.relu(
            input if input.is_mkldnn else input.to_mkldnn(),
            inplace=False,
        )
        if output.is_mkldnn:
            return output
        return output.to_mkldnn()


class _ODNNDenseBoundary(nn.Module):
    """Run an unsupported module after converting its input to dense."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return self.module(prepack_backend.to_dense(input))
