# oneDNN Extension Demo

This directory demonstrates an IPEX-like frontend at a small scale.

## Installation

Release wheels contain the compiled PyTorch extension and a private copy of
the oneDNN shared library. End users do not need a compiler, oneDNN headers,
`libdnnl.so`, or any `DNNL_*` environment variables:

```bash
python -m pip install onednn-extension-demo
```

For a fresh Linux x86_64 CPU-only environment, install PyTorch from its CPU
index first; otherwise PyPI's default Linux PyTorch package may pull NVIDIA
runtime dependencies:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0+cpu
python -m pip install onednn-extension-demo
```

Native wheels are compiled for one PyTorch minor ABI. Install normally and let
`pip` enforce the wheel's dependency range; do not use `--no-deps` to combine a
wheel with a different PyTorch minor version.

The Linux x86_64 CI builds against PyTorch's official `2.8.0+cpu` wheel, so it
does not download or bundle CUDA/NVIDIA runtime packages. Linux aarch64 uses
the native CPU-only PyTorch 2.8.0 wheel from PyPI.

To install a wheel downloaded from the GitHub Actions artifact, choose the file
whose CPython and architecture tags match the target interpreter. For example,
use a `cp312-...-x86_64.whl` with CPython 3.12 on x86_64, or a
`cp312-...-aarch64.whl` with CPython 3.12 on aarch64.

On Linux x86_64:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0+cpu
python -m pip install ./onednn_extension_demo-0.1.1-cp312-cp312-manylinux_2_28_x86_64.whl
```

On Linux aarch64, including Kunpeng-class hosts:

```bash
python -m pip install torch==2.8.0
python -m pip install ./onednn_extension_demo-0.1.1-cp312-cp312-manylinux_2_28_aarch64.whl
```

Replace `0.1.1` and `cp312` with the downloaded wheel's actual version and
Python tag. Do not rename wheel tags. No oneDNN installation or `DNNL_*` /
`LD_LIBRARY_PATH` configuration is required. Verify the installation with:

```bash
python - <<'PY'
from importlib.metadata import version

import torch
import oneDNN_extension_demo as odnn

print("package:", version("onednn-extension-demo"))
print("torch:", torch.__version__)
print("optimize available:", callable(odnn.optimize))
PY
```

The runtime loader prefers the bundled `oneDNN_extension_demo._C` library. A
source checkout without `_C` keeps the developer-oriented JIT build fallback,
which still requires a compiler and optionally `DNNL_ROOT` for native oneDNN.

The public API is:

```python
import oneDNN_extension_demo as odnn

model.eval()
model = odnn.optimize(model)
```

Pass `replace_modules=False` to benchmark layout conversion without replacing
modules:

```python
model = odnn.optimize(model, channels_last=True, replace_modules=False)
```

Pass `weight_prepack=True` to use the separate prepacked path. Conv2d and Linear
weights are converted to oneDNN/MKLDNN packed modules during `optimize()` and
are reused during inference:

```python
model = odnn.optimize(model, channels_last=True, weight_prepack=True)
```

Provide a representative input to record each Conv2d/Linear input shape and
warm the corresponding oneDNN primitive cache during optimization:

```python
model = odnn.optimize(
    model,
    channels_last=True,
    weight_prepack=True,
    sample_input=x,
)
```

In this standalone demo, `sample_input` records shape metadata and warms the
PyTorch oneDNN primitive cache. It does not query IPEX's private ideep APIs for
a shape-specific weight descriptor.

Set `cpp_op_context=True` to build and use the optional C++ CustomClassHolder
contexts. This requires a compiler compatible with the installed PyTorch:

```python
model = odnn.optimize(
    model,
    channels_last=True,
    weight_prepack=True,
    sample_input=x,
    cpp_op_context=True,
)
```

The C++ contexts own references to the packed MKLDNN weights and execute ATen's
strided convolution/linear kernels by default. They demonstrate the IPEX-style
context lifecycle without depending on IPEX's private C++ implementation.

To enable persistent native oneDNN Conv2d packing, point the build at a oneDNN
development installation before the extension is loaded:

```bash
export DNNL_ROOT=/opt/oneDNN
# Or set DNNL_INCLUDE_DIR and DNNL_LIBRARY separately.
export ODNN_DEMO_REQUIRE_NATIVE_DNNL=1
python example.py
```

With a representative `sample_input`, supported FP32 Conv2d contexts create a
shape-specific primitive and reorder weights once. A different input shape or
unsupported configuration safely uses the strided ATen fallback. Linear uses
the strided ATen path in the current native extension.

The prepacked replacement table is:

```text
nn.Conv2d -> _ODNNPrepackedConv2d
          -> ODNNConvolutionOpContext -> MkldnnConv2d
nn.Linear -> _ODNNPrepackedLinear
          -> ODNNLinearOpContext -> MkldnnLinear
```

This path prepackages weights only. By default each wrapper converts its output
back to dense layout, which is safe for eager models containing residual adds,
concatenation, resizing, splitting, or other operations that do not accept
opaque MKLDNN tensors.

For a known sequential Conv2d/BatchNorm2d/ReLU graph, continuous opaque layout
can be enabled explicitly:

```python
model = odnn.optimize(
    model,
    weight_prepack=True,
    preserve_mkldnn_layout=True,
)
```

This experimental mode inserts a dense boundary for AdaptiveAvgPool2d, but is
not graph-safe for arbitrary models. It cannot be combined with the strided
`cpp_op_context=True` path.

By default, adjacent Conv2d+BatchNorm2d and Linear+BatchNorm pairs are folded
before replacement. Disable these independently with `conv_bn_folding=False`
or `linear_bn_folding=False`.

Lazy graph capture and CPU mixed precision are also available:

```python
model = odnn.optimize(
    model,
    graph_mode=True,
    dtype=torch.bfloat16,
)
```

The first forward attempts JIT trace/freeze, then Dynamo, and finally a safe
eager fallback. The selected method is recorded by the graph wrapper after the
first call.

The default replacement path does not call `torch.utils.mkldnn.to_mkldnn()`.
It uses an explicit module replacement table:

```text
nn.Conv2d      -> _ODNNConv2d
nn.Linear      -> _ODNNLinear
nn.BatchNorm2d -> _ODNNBatchNorm2d
nn.ReLU        -> _ODNNReLU
```

The replacement modules call functions in `backend.py`. In this demo those
functions explicitly convert supported tensors to PyTorch MKLDNN tensors so
Conv2d and Linear dispatch to oneDNN/MKLDNN kernels, then convert the outputs
back to dense tensors. The separate `weight_prepack=True` path uses
`torch.utils.mkldnn.to_mkldnn()` to reorder Conv2d and Linear weights once.
For a Kunpeng/kblas extension, keep the same frontend and module wrappers but
replace the backend-specific implementation with registered Kunpeng operators.

Run:

```bash
python example.py
```

On Ubuntu/Debian, install the C++ build tools before enabling
`cpp_op_context=True`:

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build
python -m pip install ninja
```

Use a fresh Linux extension cache and enable compiler output when validating the
C++ path:

```bash
export TORCH_EXTENSIONS_DIR=/tmp/odnn_demo_extensions
export ODNN_DEMO_CPP_VERBOSE=1
python example.py
```

The default and Python prepack paths remain Python-only. The optional C++
OpContext path JIT-compiles `csrc/op_context.cpp` into a Linux `.so` on first
use. This demo mirrors the frontend structure needed for a later backend:
build a lookup table, replace supported modules, then execute backend-specific
kernels.

## Building Release Wheels

The `Build native wheels` GitHub Actions workflow builds oneDNN from a pinned
tag, compiles `_C` against the selected PyTorch version, and runs
`auditwheel repair`. The repaired wheel contains `libdnnl.so` under its private
library directory and uses a relative runtime search path.

Run the workflow manually to validate artifacts before publishing. Choose the
PyTorch version that the release supports, such as `2.8.0`; the generated wheel
metadata constrains installation to `torch>=2.8,<2.9`.

For a local Linux build, install the target PyTorch version and build without
PEP 517 isolation so the extension uses that exact ABI:

```bash
export ODNN_BUILD_NATIVE=1
export DNNL_ROOT=/opt/onednn
python -m pip wheel . --no-build-isolation -w wheelhouse
auditwheel repair wheelhouse/*.whl -w repaired-wheelhouse
```

Publishing a GitHub Release runs the same build and uploads the repaired wheels
to PyPI through Trusted Publishing. Before the first release, configure a PyPI
Trusted Publisher for this repository and the workflow
`.github/workflows/build-wheels.yml`. No long-lived PyPI token is required.

### Fast Python-only wheel repacking

After a successful native build, the `Repack Python-only wheels` workflow can
reuse the repaired `_C.so` and bundled `libdnnl*.so` from that run. It replaces
the package's Python sources, assigns a new package version, rebuilds `RECORD`,
and uploads the repacked wheels without compiling PyTorch C++ or oneDNN again.

Open the successful `Build native wheels` run and copy the numeric ID from its
URL (`.../actions/runs/<run-id>`). Then open **Actions**, select
**Repack Python-only wheels**, choose **Run workflow**, and enter:

- `native_run_id`: the successful native build run ID.
- `package_version`: a new PEP 440 version, such as `0.1.1`.
- `smoke_test_x86_64`: optionally install `torch==2.8.0+cpu` and run the
  CPython 3.12 x86_64 smoke test. This takes longer but does not install CUDA.

Download the `python-only-wheels-<version>` artifact after the job completes.
The workflow repacks every wheel found in both x86_64 and aarch64 artifacts.
It does not publish them to PyPI automatically.

This workflow intentionally rejects changes to C++ sources, package data,
licenses, `setup.py`, `pyproject.toml`, `MANIFEST.in`, or
`tools/build_onednn.sh`. Changes to the PyTorch version, oneDNN version,
compiler flags, native source, Python ABI matrix, or target architecture require
running `Build native wheels`.

The initial build matrix publishes CPython 3.9-3.12 wheels for Linux x86_64
and Linux aarch64, including Kunpeng-class hosts. Other operating systems need
separate native wheel jobs; an x86_64 `libdnnl.so` must never be reused on an
aarch64 platform, or vice versa.
