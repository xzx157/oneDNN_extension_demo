import copy
import warnings
from dataclasses import dataclass
from typing import Dict, Type

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils.fusion import fuse_linear_bn_eval

from .modules import _ODNNBatchNorm2d, _ODNNConv2d, _ODNNLinear, _ODNNReLU
from .prepack_modules import (
    _ODNNMKLDNNBatchNorm2d,
    _ODNNMKLDNNReLU,
    _ODNNPrepackedConv2d,
    _ODNNPrepackedLinear,
    _ODNNDenseBoundary,
)
from .graph_mode import GraphCaptureLite, _RunMethod


MODULE_REPLACEMENT_TABLE: Dict[Type[nn.Module], Type[nn.Module]] = {
    nn.Conv2d: _ODNNConv2d,
    nn.Linear: _ODNNLinear,
    nn.BatchNorm2d: _ODNNBatchNorm2d,
    nn.ReLU: _ODNNReLU,
}

WEIGHT_PREPACK_REPLACEMENT_TABLE: Dict[Type[nn.Module], Type[nn.Module]] = {
    nn.Conv2d: _ODNNPrepackedConv2d,
    nn.Linear: _ODNNPrepackedLinear,
    nn.BatchNorm2d: _ODNNMKLDNNBatchNorm2d,
    nn.ReLU: _ODNNMKLDNNReLU,
    nn.AdaptiveAvgPool2d: _ODNNDenseBoundary,
}


@dataclass
class OptimizeReport:
    total_modules: int
    replaced_modules: int
    folded_batchnorms: int
    channels_last: bool
    backend: str = "oneDNN-demo"
    replace_modules: bool = True
    weight_prepack: bool = False
    conv_bn_folding: bool = True
    linear_bn_folding: bool = True
    sample_input: bool = False
    cpp_op_context: bool = False
    graph_mode: bool = False
    capture_method: str = ""


def optimize(
    model: nn.Module,
    *,
    inplace: bool = False,
    channels_last: bool = True,
    replace_modules: bool = True,
    weight_prepack: bool = False,
    conv_bn_folding: bool = True,
    linear_bn_folding: bool = True,
    sample_input=None,
    cpp_op_context: bool = False,
    dtype=None,
    graph_mode: bool = False,
    return_report: bool = False,
):
    """Apply layout conversion and optional oneDNN module replacement.

    weight_prepack=False keeps the original explicit backend path.
    weight_prepack=True converts Conv2d and Linear modules with
    torch.utils.mkldnn.to_mkldnn() during optimize(), so their weights are
    reordered once and reused by subsequent forward calls.

    conv_bn_folding=True folds Conv+BatchNorm pairs in eval mode before module
    replacement and optional prepack warmup.

    linear_bn_folding=True folds Linear+BatchNorm(1d/2d/3d) pairs in eval mode
    using FX pattern matching before module replacement.

    graph_mode=True wraps model.forward with lazy JIT graph capture
    (JIT trace+freeze → dynamo → eager fallback).  dtype=torch.bfloat16
    or torch.float16 enables CPU autocast for mixed-precision inference.
    """

    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("PyTorch was built without MKLDNN/oneDNN support.")
    if weight_prepack and not replace_modules:
        raise ValueError("weight_prepack=True requires replace_modules=True.")
    if cpp_op_context and not weight_prepack:
        raise ValueError("cpp_op_context=True requires weight_prepack=True.")

    opt_model = model if inplace else copy.deepcopy(model)
    opt_model.eval()

    folded_batchnorms = 0
    if conv_bn_folding:
        opt_model, folded_batchnorms = _fold_conv_bn(opt_model)
    if linear_bn_folding:
        opt_model, folded_linear_bn = _fold_linear_bn(opt_model)
        folded_batchnorms += folded_linear_bn

    if channels_last:
        _convert_conv_weight_to_channels_last(opt_model)
    if sample_input is not None:
        _record_sample_input_sizes(opt_model, sample_input)

    stats = {"total": 0, "replaced": 0}
    if replace_modules:
        replacement_table = (
            WEIGHT_PREPACK_REPLACEMENT_TABLE
            if weight_prepack
            else MODULE_REPLACEMENT_TABLE
        )
        opt_model = _replace_modules(
            opt_model,
            channels_last=channels_last,
            replacement_table=replacement_table,
            cpp_op_context=cpp_op_context,
            stats=stats,
        )
    else:
        stats["total"] = sum(1 for _ in opt_model.modules())

    if weight_prepack and sample_input is not None:
        _warmup_model(opt_model, sample_input)

    capture_method = ""
    if graph_mode:
        wrapper = GraphCaptureLite(opt_model, dtype=dtype)
        opt_model.forward = wrapper(opt_model.forward)
        capture_method = _RunMethod.label(wrapper.method)

    report = OptimizeReport(
        total_modules=stats["total"],
        replaced_modules=stats["replaced"],
        folded_batchnorms=folded_batchnorms,
        channels_last=channels_last,
        replace_modules=replace_modules,
        weight_prepack=weight_prepack,
        conv_bn_folding=conv_bn_folding,
        linear_bn_folding=linear_bn_folding,
        sample_input=sample_input is not None,
        cpp_op_context=cpp_op_context,
        graph_mode=graph_mode,
        capture_method=capture_method,
        backend=(
            "oneDNN-cpp-prepack"
            if cpp_op_context
            else "oneDNN-prepack"
            if weight_prepack
            else "oneDNN-demo"
        ),
    )
    opt_model.optimize_report = report

    if return_report:
        return opt_model, report
    return opt_model


def _replace_modules(
    module: nn.Module,
    *,
    channels_last: bool,
    replacement_table,
    cpp_op_context: bool,
    stats,
) -> nn.Module:
    stats["total"] += 1
    replacement_cls = replacement_table.get(module.__class__)
    if replacement_cls is not None:
        stats["replaced"] += 1
        if replacement_cls in (_ODNNConv2d, _ODNNPrepackedConv2d):
            if replacement_cls is _ODNNPrepackedConv2d:
                return replacement_cls(
                    module,
                    channels_last=channels_last,
                    cpp_op_context=cpp_op_context,
                )
            return replacement_cls(module, channels_last=channels_last)
        if replacement_cls is _ODNNPrepackedLinear:
            return replacement_cls(module, cpp_op_context=cpp_op_context)
        return replacement_cls(module)

    for name, child in list(module.named_children()):
        setattr(
            module,
            name,
            _replace_modules(
                child,
                channels_last=channels_last,
                replacement_table=replacement_table,
                cpp_op_context=cpp_op_context,
                stats=stats,
            ),
        )
    return module


def _convert_conv_weight_to_channels_last(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            child.weight.data = child.weight.detach().clone().contiguous(
                memory_format=torch.channels_last
            )


def _count_batchnorm_modules(module: nn.Module) -> int:
    return sum(1 for child in module.modules() if isinstance(child, _BatchNorm))


def _fold_conv_bn(module: nn.Module):
    """Fold Conv2d+BatchNorm2d in eval mode by walking the module tree.

    Unlike ``optimization.fuse()``, this does NOT use FX symbolic_trace and
    therefore works on models with dynamic control flow (e.g. YOLO).
    """

    before = _count_batchnorm_modules(module)
    _fuse_adjacent_pairs(
        module,
        leader_cls=nn.Conv2d,
        follower_cls=nn.BatchNorm2d,
        fuse_fn=_fuse_single_conv_bn,
    )
    after = _count_batchnorm_modules(module)
    return module, max(before - after, 0)


def _fold_linear_bn(module: nn.Module):
    """Fold Linear+BatchNorm in eval mode by walking the module tree."""

    before = _count_batchnorm_modules(module)
    for bn_cls in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
        _fuse_adjacent_pairs(
            module,
            leader_cls=nn.Linear,
            follower_cls=bn_cls,
            fuse_fn=_fuse_single_linear_bn,
        )
    after = _count_batchnorm_modules(module)
    return module, max(before - after, 0)


def _fuse_adjacent_pairs(parent, *, leader_cls, follower_cls, fuse_fn):
    """Walk the module tree and fuse consecutive leader→follower children in-place.

    This operates on the *immediate children* of every submodule, so it handles
    nn.Sequential and custom containers equally well without any tracing.
    """

    for child in parent.children():
        _fuse_adjacent_pairs(child, leader_cls=leader_cls, follower_cls=follower_cls, fuse_fn=fuse_fn)

    # OrderedDict preserves insertion order → matches execution order
    names = list(parent._modules.keys())
    i = 0
    while i < len(names) - 1:
        cur = parent._modules[names[i]]
        nxt = parent._modules[names[i + 1]]
        if isinstance(cur, leader_cls) and isinstance(nxt, follower_cls):
            if not getattr(nxt, "track_running_stats", True):
                i += 2
                continue
            try:
                fused = fuse_fn(cur, nxt)
            except Exception:
                i += 2
                continue
            parent._modules[names[i]] = fused
            parent._modules[names[i + 1]] = nn.Identity()
            names.pop(i + 1)
        else:
            i += 1


def _fuse_single_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    return torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)


def _fuse_single_linear_bn(linear: nn.Linear, bn: _BatchNorm):
    return fuse_linear_bn_eval(linear, bn)


def _record_sample_input_sizes(model: nn.Module, sample_input) -> None:
    if isinstance(sample_input, torch.Tensor):
        sample_input = (sample_input,)
    elif not isinstance(sample_input, tuple):
        sample_input = tuple(sample_input)

    hooks = []

    def record_input(module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            module._odnn_sample_input_size = tuple(inputs[0].shape)

    for child in model.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            hooks.append(child.register_forward_pre_hook(record_input))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(*sample_input)
    finally:
        if was_training:
            model.train()
        for hook in hooks:
            hook.remove()


def _warmup_model(model: nn.Module, sample_input) -> None:
    if isinstance(sample_input, torch.Tensor):
        sample_input = (sample_input,)
    elif not isinstance(sample_input, tuple):
        sample_input = tuple(sample_input)
    with torch.inference_mode():
        model(*sample_input)


def inspect_modules(model: nn.Module) -> Dict[str, str]:
    return {name: module.__class__.__name__ for name, module in model.named_modules()}
