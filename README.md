# oneDNN Extension Demo

This directory demonstrates an IPEX-like frontend at a small scale.

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
MKLDNN convolution/linear kernels. They demonstrate the IPEX-style context
lifecycle without depending on IPEX's private C++ implementation.

The prepacked replacement table is:

```text
nn.Conv2d -> _ODNNPrepackedConv2d
          -> ODNNConvolutionOpContext -> MkldnnConv2d
nn.Linear -> _ODNNPrepackedLinear
          -> ODNNLinearOpContext -> MkldnnLinear
```

This path prepackages weights only. Activations are still converted to MKLDNN
at the first prepacked wrapper boundary. Conv2d, BatchNorm2d, and ReLU keep the
MKLDNN layout across consecutive operations. Unsupported boundaries such as
AdaptiveAvgPool2d convert back to dense, and Linear returns a dense output.

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
python -m oneDNN_extension_demo.example
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
python -m oneDNN_extension_demo.example
```

The default and Python prepack paths remain Python-only. The optional C++
OpContext path JIT-compiles `csrc/op_context.cpp` into a Linux `.so` on first
use. This demo mirrors the frontend structure needed for a later backend:
build a lookup table, replace supported modules, then execute backend-specific
kernels.
