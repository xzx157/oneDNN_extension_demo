import os
import time

import torch
from torch import nn

import oneDNN_extension_demo as odnn
from oneDNN_extension_demo.frontend import inspect_modules
from oneDNN_extension_demo.prepack_context import ODNNConvolutionOpContext


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 16)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def benchmark(model, x, *, warmup=20, iters=100):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        elapsed = time.perf_counter() - start

    return elapsed / iters * 1000.0


def report_speedup(name, baseline_ms, optimized_ms):
    print(f"{name}: {optimized_ms:.3f} ms/iter")
    print(f"{name} speedup vs baseline: {baseline_ms / optimized_ms:.2f}x")


def create_ipex_o1_model(model, sample_input):
    try:
        import intel_extension_for_pytorch as ipex

        return (
            ipex.optimize(
                model,
                level="O1",
                inplace=False,
                sample_input=sample_input,
            ),
            None,
        )
    except (ImportError, OSError, RuntimeError) as error:
        return None, f"{type(error).__name__}: {error}"


def create_ipex_prepack_model(model, sample_input):
    try:
        import intel_extension_for_pytorch as ipex

        return (
            ipex.optimize(
                model,
                level="O0",
                inplace=False,
                weights_prepack=True,
                sample_input=sample_input,
            ),
            None,
        )
    except (ImportError, OSError, RuntimeError, TypeError) as error:
        return None, f"{type(error).__name__}: {error}"


def collect_cpp_runtime_stats(model):
    stats = {"native_runs": 0, "fallback_runs": 0, "native_hit_rate": 0.0}
    for module in model.modules():
        if not isinstance(module, ODNNConvolutionOpContext):
            continue
        module_stats = module.runtime_stats()
        stats["native_runs"] += module_stats["native_runs"]
        stats["fallback_runs"] += module_stats["fallback_runs"]

    total_runs = stats["native_runs"] + stats["fallback_runs"]
    if total_runs:
        stats["native_hit_rate"] = stats["native_runs"] / total_runs
    return stats


def main():
    torch.manual_seed(0)

    num_threads = os.environ.get("ODNN_DEMO_NUM_THREADS")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    model = TinyCNN().eval()
    x = torch.randn(32, 3, 224, 224)
    x_channels_last = x.contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        ref = model(x)

    channels_last_model, channels_last_report = odnn.optimize(
        model,
        channels_last=True,
        replace_modules=False,
        return_report=True,
    )
    try:
        prepacked_model, prepacked_report = odnn.optimize(
            model,
            channels_last=True,
            weight_prepack=True,
            sample_input=x_channels_last,
            return_report=True,
        )
        prepacked_error = None
    except (ImportError, OSError, RuntimeError) as error:
        prepacked_model = None
        prepacked_report = None
        prepacked_error = f"{type(error).__name__}: {error}"
    ipex_o1_model, ipex_o1_error = create_ipex_o1_model(model, x_channels_last)
    ipex_prepack_model, ipex_prepack_error = create_ipex_prepack_model(
        model, x_channels_last
    )
    opt_model, report = odnn.optimize(model, channels_last=True, return_report=True)
    opt_model_nchw, report_nchw = odnn.optimize(
        model, channels_last=False, return_report=True
    )

    with torch.inference_mode():
        out_channels_last = channels_last_model(x_channels_last)
        out_prepacked = (
            prepacked_model(x_channels_last) if prepacked_model is not None else None
        )
        out = opt_model(x)
        out_nchw = opt_model_nchw(x)
        out_ipex_o1 = (
            ipex_o1_model(x_channels_last) if ipex_o1_model is not None else None
        )
        out_ipex_prepack = (
            ipex_prepack_model(x_channels_last)
            if ipex_prepack_model is not None
            else None
        )

    baseline_ms = benchmark(model, x)
    channels_last_ms = benchmark(channels_last_model, x_channels_last)
    prepacked_ms = (
        benchmark(prepacked_model, x_channels_last)
        if prepacked_model is not None
        else None
    )
    odnn_ms = benchmark(opt_model, x)
    odnn_nchw_ms = benchmark(opt_model_nchw, x)
    ipex_o1_ms = (
        benchmark(ipex_o1_model, x_channels_last)
        if ipex_o1_model is not None
        else None
    )
    ipex_prepack_ms = (
        benchmark(ipex_prepack_model, x_channels_last)
        if ipex_prepack_model is not None
        else None
    )

    print("Channels-last-only report:", channels_last_report)
    print("Weight-prepack report:", prepacked_report)
    print("Optimize report:", report)
    print("Optimize report without channels_last:", report_nchw)
    print(
        "Max abs diff channels-last only:",
        (ref - out_channels_last).abs().max().item(),
    )
    if out_prepacked is not None:
        print(
            "Max abs diff C++ OpContext:",
            (ref - out_prepacked).abs().max().item(),
        )
        print(
            "C++ OpContext runtime stats:",
            collect_cpp_runtime_stats(prepacked_model),
        )
    else:
        print("C++ OpContext skipped:", prepacked_error)
    print("Max abs diff:", (ref - out).abs().max().item())
    print("Max abs diff without channels_last:", (ref - out_nchw).abs().max().item())
    if out_ipex_o1 is not None:
        print("Max abs diff IPEX O1:", (ref - out_ipex_o1).abs().max().item())
    else:
        print("IPEX O1 skipped:", ipex_o1_error)
    if out_ipex_prepack is not None:
        print(
            "Max abs diff IPEX O0 + weight prepack:",
            (ref - out_ipex_prepack).abs().max().item(),
        )
    else:
        print("IPEX O0 + weight prepack skipped:", ipex_prepack_error)
    print("Benchmark input:", tuple(x.shape))
    print(f"Baseline PyTorch: {baseline_ms:.3f} ms/iter")
    report_speedup("Channels-last only", baseline_ms, channels_last_ms)
    if prepacked_ms is not None:
        report_speedup("C++ OpContext + sample_input", baseline_ms, prepacked_ms)
    report_speedup("ODNN explicit MKLDNN + channels_last", baseline_ms, odnn_ms)
    report_speedup("ODNN explicit MKLDNN no channels_last", baseline_ms, odnn_nchw_ms)
    if ipex_o1_ms is not None:
        report_speedup("IPEX O1", baseline_ms, ipex_o1_ms)
    if ipex_prepack_ms is not None:
        report_speedup(
            "IPEX O0 + weight prepack",
            baseline_ms,
            ipex_prepack_ms,
        )
    print("Converted modules:")
    for name, class_name in inspect_modules(opt_model).items():
        if class_name.startswith("_ODNN"):
            print(f"  {name}: {class_name}")


if __name__ == "__main__":
    main()
