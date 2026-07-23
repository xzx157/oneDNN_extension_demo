"""注册 odnn::* 自定义算子。

这些算子被 modules.py 中的 _ODNNConv2d / _ODNNLinear / _ODNNReLU / _ODNNBatchNorm2d
调用，实现通过 backend.py 转发到 PyTorch MKLDNN。
"""

import torch


# 创建 odnn 库并定义算子 schema
odnn_lib = torch.library.Library("odnn", "DEF")

odnn_lib.define(
    "conv2d(Tensor input, Tensor weight, Tensor bias, "
    "int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"
)
odnn_lib.define(
    "linear(Tensor input, Tensor weight, Tensor bias) -> Tensor"
)
odnn_lib.define(
    "relu(Tensor input, bool inplace=False) -> Tensor"
)
odnn_lib.define(
    "batchnorm2d(Tensor input, Tensor running_mean, Tensor running_var, "
    "Tensor weight, Tensor bias, bool training, float momentum, float eps) -> Tensor"
)


# 注册 CPU 实现
@torch.library.impl(odnn_lib, "conv2d", "CPU")
def conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    dilation,
    groups,
) -> torch.Tensor:
    from . import backend
    return backend.conv2d(input, weight, bias, stride, padding, dilation, groups)


@torch.library.impl(odnn_lib, "linear", "CPU")
def linear_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    from . import backend
    return backend.linear(input, weight, bias)


@torch.library.impl(odnn_lib, "relu", "CPU")
def relu_impl(
    input: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    from . import backend
    return backend.relu(input, inplace)


@torch.library.impl(odnn_lib, "batchnorm2d", "CPU")
def batchnorm2d_impl(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> torch.Tensor:
    from . import backend
    return backend.batch_norm2d(
        input, running_mean, running_var, weight, bias,
        training, momentum, eps,
    )
