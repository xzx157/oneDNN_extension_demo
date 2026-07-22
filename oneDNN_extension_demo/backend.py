import torch
import torch.nn.functional as F


def _require_mkldnn():
    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("PyTorch was built without MKLDNN/oneDNN support.")
    if not torch.backends.mkldnn.enabled:
        raise RuntimeError("PyTorch MKLDNN/oneDNN backend is disabled.")


def _to_dense(output):
    if isinstance(output, torch.Tensor) and output.is_mkldnn:
        return output.to_dense()
    return output


def _to_mkldnn(input):
    if isinstance(input, torch.Tensor) and input.is_mkldnn:
        return input
    return input.to_mkldnn()


def conv2d(input, weight, bias, stride, padding, dilation, groups):
    """oneDNN demo backend entry for Conv2d.

    Convert inputs to MKLDNN tensors so PyTorch dispatches this convolution to
    its oneDNN/MKLDNN implementation, then return a normal dense tensor so the
    rest of the model can keep using ordinary PyTorch modules.
    """

    _require_mkldnn()
    output = F.conv2d(
        _to_mkldnn(input),
        _to_mkldnn(weight),
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
    return _to_dense(output)


def linear(input, weight, bias):
    """oneDNN demo backend entry for Linear."""

    _require_mkldnn()
    input_mkldnn = _to_mkldnn(input)
    weight_mkldnn = _to_mkldnn(weight)

    if hasattr(torch._C._nn, "mkldnn_linear"):
        output = torch._C._nn.mkldnn_linear(input_mkldnn, weight_mkldnn, bias)
    else:
        output = torch.ops.aten.mkldnn_linear(input_mkldnn, weight_mkldnn, bias)

    return _to_dense(output)


def relu(input, inplace=False):
    _require_mkldnn()
    output = F.relu(_to_mkldnn(input), inplace=inplace)
    return _to_dense(output)


def batch_norm2d(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    _require_mkldnn()
    output = F.batch_norm(
        _to_mkldnn(input),
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
    )
    return _to_dense(output)


def ensure_channels_last(input):
    if isinstance(input, torch.Tensor) and input.dim() == 4:
        if not input.is_contiguous(memory_format=torch.channels_last):
            return input.contiguous(memory_format=torch.channels_last)
    return input
