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

Pass `weight_prepack=True` to use the C++ OpContext prepacked path. Conv2d and
Linear weights are captured by C++ context objects during `optimize()` and are
reused during inference:

```python
model = odnn.optimize(model, channels_last=True, weight_prepack=True)
```

Provide a representative input to record each Conv2d/Linear input shape and
warm the corresponding backend path during optimization:

```python
model = odnn.optimize(
    model,
    channels_last=True,
    weight_prepack=True,
    sample_input=x,
)
```

In this standalone demo, `sample_input` records shape metadata for the C++
contexts. It does not query IPEX's private ideep APIs for a shape-specific
weight descriptor.

The C++ contexts own references to the packed weights and execute native oneDNN
Conv2d when available, with a strided ATen fallback. Linear uses the strided
ATen path in the current native extension. This demonstrates the IPEX-style
context lifecycle without depending on IPEX's private C++ implementation.

To enable persistent native oneDNN Conv2d packing, point the build at a oneDNN
development installation before the extension is loaded:

```bash
export DNNL_ROOT=/opt/oneDNN
# Or set DNNL_INCLUDE_DIR and DNNL_LIBRARY separately.
export ODNN_DEMO_REQUIRE_NATIVE_DNNL=1
python example.py
```

### Download prebuilt oneDNN (no compilation required)

The `third_party/onednn/` directory already contains a prebuilt oneDNN v3.8
(OMP runtime, CPU-only, no SYCL/GPU dependency). To refresh or obtain it
independently:

```bash
# 1. Download from conda-forge (OMP variant, no DPC++):
#    For faster downloads in China, use the Tsinghua mirror.
curl -L --retry 3 -o /tmp/onednn-3.8.1-omp.conda \
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/onednn-3.8.1-omp_hff5f129_1.conda"
#    Or the official source:
# curl -L --retry 3 -o /tmp/onednn-3.8.1-omp.conda \
#   "https://api.anaconda.org/download/conda-forge/onednn/3.8.1/linux-64/onednn-3.8.1-omp_hff5f129_1.conda"

# 2. Extract (requires zstandard):
pip install zstandard
python3 - << 'PYEOF'
import zipfile, zstandard, io, tarfile, os, shutil

extract_dir = "/tmp/onednn-extract"
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir)

with zipfile.ZipFile("/tmp/onednn-3.8.1-omp.conda") as z:
    data = z.read([n for n in z.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")][0])

decompressed = zstandard.ZstdDecompressor().decompress(data)
with tarfile.open(fileobj=io.BytesIO(decompressed)) as tar:
    tar.extractall(extract_dir)

print("Extracted to", extract_dir)
PYEOF

# 3. Copy to third_party/:
rm -rf third_party/onednn
mkdir -p third_party/onednn/lib third_party/onednn/include
cp -a /tmp/onednn-extract/lib/libdnnl.so* third_party/onednn/lib/
cp -a /tmp/onednn-extract/include/* third_party/onednn/include/

# 4. Verify:
ls third_party/onednn/lib/libdnnl.so.3.8 && echo "oneDNN OK"
```

The OMP runtime variant depends only on system libraries (`libgomp`, `libstdc++`,
etc.) and does **not** require SYCL, OpenCL, TBB, or a GPU driver. Choose the
`omp_*` build tag in the conda-forge filename; avoid `dpcpp_*` variants unless
you have a working oneAPI DPC++/SYCL runtime.

With a representative `sample_input`, supported FP32 Conv2d contexts create a
shape-specific primitive and reorder weights once. A different input shape or
unsupported configuration safely uses the strided ATen fallback. Linear uses
the strided ATen path in the current native extension.

The prepacked replacement table is:

```text
nn.Conv2d -> _ODNNPrepackedConv2d
          -> ODNNConvolutionOpContext -> C++ ConvolutionOpContext
nn.Linear -> _ODNNPrepackedLinear
          -> ODNNLinearOpContext -> C++ LinearOpContext
```

This path keeps eager tensors in dense/strided layout, which is safe for models
containing residual adds, concatenation, resizing, splitting, or other ordinary
PyTorch tensor operations.

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
back to dense tensors. The separate `weight_prepack=True` path creates C++
OpContext wrappers for Conv2d and Linear.
For a Kunpeng/kblas extension, keep the same frontend and module wrappers but
replace the backend-specific implementation with registered Kunpeng operators.

Run:

```bash
python example.py
```

On Ubuntu/Debian, install the C++ build tools before using a source checkout
with `weight_prepack=True`:

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

The default replacement path remains Python-only. The prepack path JIT-compiles
`csrc/op_context.cpp` into a Linux `.so` on first use when a prebuilt `_C`
extension is unavailable. This demo mirrors the frontend structure needed for a
later backend: build a lookup table, replace supported modules, then execute
backend-specific kernels.

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

The initial build matrix publishes CPython 3.9-3.12 wheels for Linux x86_64
and Linux aarch64, including Kunpeng-class hosts. Other operating systems need
separate native wheel jobs; an x86_64 `libdnnl.so` must never be reused on an
aarch64 platform, or vice versa.

问题根因：torch.ops.load_library() 底层 dlopen 在 import 早期严格验证 NEEDED 库路径。已改用 ctypes.CDLL + RTLD_GLOBAL 来避免。
