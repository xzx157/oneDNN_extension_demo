from torch import nn

from .prepack_context import ODNNConvolutionOpContext, ODNNLinearOpContext


class _ODNNPrepackedConv2d(nn.Module):
    def __init__(
        self,
        module,
        *,
        channels_last=False,
    ):
        super().__init__()
        self.ctx = ODNNConvolutionOpContext.create_context(
            module,
            channels_last=channels_last,
            input_size=getattr(module, "_odnn_sample_input_size", ()),
        )

    def forward(self, input):
        return self.ctx.run(input)


class _ODNNPrepackedLinear(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.ctx = ODNNLinearOpContext.create_context(
            module,
            input_size=getattr(module, "_odnn_sample_input_size", ()),
        )

    def forward(self, input):
        return self.ctx.run(input)
