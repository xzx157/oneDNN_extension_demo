from torch import nn

from . import backend


class _ODNNConv2d(nn.Module):
    def __init__(self, module, *, channels_last=False):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.channels_last = channels_last

        if self.padding_mode != "zeros":
            self._reversed_padding_repeated_twice = (
                module._reversed_padding_repeated_twice
            )

    def forward(self, input):
        if self.channels_last:
            input = backend.ensure_channels_last(input)

        if self.padding_mode != "zeros":
            from torch.nn import functional as F

            input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            padding = 0
        else:
            padding = self.padding

        return backend.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )


class _ODNNLinear(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias
        self.in_features = module.in_features
        self.out_features = module.out_features

    def forward(self, input):
        return backend.linear(input, self.weight, self.bias)


class _ODNNReLU(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.inplace = module.inplace

    def forward(self, input):
        return backend.relu(input, self.inplace)


class _ODNNBatchNorm2d(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias
        self.running_mean = module.running_mean
        self.running_var = module.running_var
        self.training = module.training
        self.momentum = module.momentum
        self.eps = module.eps

    def forward(self, input):
        return backend.batch_norm2d(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
