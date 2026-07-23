import os
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
BUILD_NATIVE = os.environ.get("ODNN_BUILD_NATIVE") == "1"
PACKAGE_VERSION = os.environ.get("ODNN_PACKAGE_VERSION", "0.1.0").lstrip("v")


def _native_build_config():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
    except ImportError as error:
        raise RuntimeError(
            "ODNN_BUILD_NATIVE=1 requires PyTorch in the build environment. "
            "Install the target PyTorch version and build without isolation."
        ) from error

    root = os.environ.get("DNNL_ROOT")
    include_dir = os.environ.get("DNNL_INCLUDE_DIR")
    library = os.environ.get("DNNL_LIBRARY")
    if root:
        root_path = Path(root).expanduser().resolve()
        include_dir = include_dir or str(root_path / "include")
        if library is None:
            candidates = (
                [root_path / "lib" / "dnnl.lib"]
                if os.name == "nt"
                else [
                    root_path / "lib" / "libdnnl.so",
                    root_path / "lib64" / "libdnnl.so",
                    root_path / "lib" / "libdnnl.dylib",
                ]
            )
            library = next(
                (
                    str(candidate)
                    for candidate in candidates
                    if candidate.is_file()
                ),
                None,
            )

    if not include_dir or not library:
        raise RuntimeError(
            "Native wheel builds require DNNL_ROOT or both DNNL_INCLUDE_DIR "
            "and DNNL_LIBRARY."
        )

    include_path = Path(include_dir).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    header = include_path / "oneapi" / "dnnl" / "dnnl.hpp"
    if not header.is_file():
        raise RuntimeError(f"oneDNN header not found: {header}")
    if not library_path.is_file():
        raise RuntimeError(f"oneDNN library not found: {library_path}")

    cxx_flags = ["/O2"] if os.name == "nt" else ["-O3", "-std=c++17"]
    rpath_flags = [] if os.name == "nt" else ["-Wl,-rpath,$ORIGIN"]
    extension = CppExtension(
        name="oneDNN_extension_demo._C",
        sources=["oneDNN_extension_demo/csrc/op_context.cpp"],
        include_dirs=[str(include_path)],
        define_macros=[("ODNN_DEMO_USE_DNNL", "1")],
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=[str(library_path)] + rpath_flags,
    )

    # 新增：aten 劫持扩展（dnnl::* 直接调用）
    hijack_sources = [
        "oneDNN_extension_demo/csrc/onednn_hijack.cpp",
        "oneDNN_extension_demo/csrc/kernels/eltwise.cpp",
        "oneDNN_extension_demo/csrc/kernels/pooling.cpp",
        "oneDNN_extension_demo/csrc/kernels/softmax.cpp",
        "oneDNN_extension_demo/csrc/kernels/binary.cpp",
        "oneDNN_extension_demo/csrc/kernels/unary.cpp",
        "oneDNN_extension_demo/csrc/kernels/normalization.cpp",
        "oneDNN_extension_demo/csrc/kernels/matmul.cpp",
    ]
    hijack_extension = CppExtension(
        name="oneDNN_extension_demo._C_hijack",
        sources=hijack_sources,
        include_dirs=[str(include_path)],
        define_macros=[("ODNN_DEMO_USE_DNNL", "1")],
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=[str(library_path)] + rpath_flags,
    )

    version = torch.__version__.split("+", 1)[0].split(".")
    torch_requirement = (
        f"torch>={version[0]}.{version[1]},"
        f"<{version[0]}.{int(version[1]) + 1}"
    )
    return (
        [extension, hijack_extension],
        {"build_ext": BuildExtension.with_options(use_ninja=True)},
        torch_requirement,
    )

    version = torch.__version__.split("+", 1)[0].split(".")
    torch_requirement = (
        f"torch>={version[0]}.{version[1]},"
        f"<{version[0]}.{int(version[1]) + 1}"
    )
    return (
        [extension, hijack_extension],
        {"build_ext": BuildExtension.with_options(use_ninja=True)},
        torch_requirement,
    )


if BUILD_NATIVE:
    ext_modules, cmdclass, torch_requirement = _native_build_config()
else:
    ext_modules, cmdclass, torch_requirement = [], {}, "torch>=2.0"


setup(
    version=PACKAGE_VERSION,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[torch_requirement, "typing-extensions>=4.0"],
)
