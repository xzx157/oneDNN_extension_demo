import torch


def _require_mkldnn():
    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("PyTorch was built without MKLDNN/oneDNN support.")
    if not torch.backends.mkldnn.enabled:
        raise RuntimeError("PyTorch MKLDNN/oneDNN backend is disabled.")


def _to_mkldnn(input):
    if isinstance(input, torch.Tensor) and input.is_mkldnn:
        return input
    return input.to_mkldnn()


def _to_dense(output):
    if isinstance(output, torch.Tensor) and output.is_mkldnn:
        return output.to_dense()
    return output


def to_dense(input):
    return _to_dense(input)


def prepack(module):
    """Create a PyTorch oneDNN module with its weight packed once.

    torch.utils.mkldnn.to_mkldnn() creates MkldnnLinear/MkldnnConv2d
    wrappers and reorders their weights during this call, rather than during
    every forward pass.
    """

    _require_mkldnn()
    from torch.utils import mkldnn as mkldnn_utils

    return mkldnn_utils.to_mkldnn(module)


def run(packed_module, input):
    """Run a prepacked oneDNN module and return a dense tensor."""

    _require_mkldnn()
    return _to_dense(packed_module(_to_mkldnn(input)))


def run_mkldnn(packed_module, input):
    """Run a prepacked module and keep its MKLDNN output layout."""

    _require_mkldnn()
    return packed_module(_to_mkldnn(input))
