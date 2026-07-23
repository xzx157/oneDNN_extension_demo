import tempfile
from pathlib import Path

import torch
from torch import nn

import oneDNN_extension_demo as odnn
from oneDNN_extension_demo.cpp_extension import cpp_extension_status
from oneDNN_extension_demo.prepack_context import ODNNConvolutionOpContext


def main():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 4),
    ).eval()
    sample = torch.randn(2, 3, 8, 8).contiguous(
        memory_format=torch.channels_last
    )

    with torch.inference_mode():
        expected = model(sample)
        optimized, report = odnn.optimize(
            model,
            weight_prepack=True,
            cpp_op_context=True,
            sample_input=sample,
            return_report=True,
        )
        actual = optimized(sample)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    status = cpp_extension_status()
    assert status == {"loaded": True, "source": "prebuilt-wheel"}, status
    contexts = [
        module
        for module in optimized.modules()
        if isinstance(module, ODNNConvolutionOpContext)
    ]
    assert contexts and all(context.uses_native_dnnl() for context in contexts)
    assert report.native_dnnl_contexts == len(contexts)
    assert report.packed_weight_bytes > 0

    # The repaired wheel must not depend on an externally configured oneDNN.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.txt"
        path.write_text(str(status), encoding="ascii")
        assert path.read_text(encoding="ascii")

    print(
        "wheel smoke test passed:",
        torch.__version__,
        report.backend,
        report.packed_weight_bytes,
    )


if __name__ == "__main__":
    main()
