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
            # 兜底：oneDNN cmake install 可能只生成 libdnnl.so.3 无 libdnnl.so 软链接
            if library is None and os.name != "nt":
                for lib_dir in (root_path / "lib", root_path / "lib64"):
                    so_files = sorted(lib_dir.glob("libdnnl.so.*"), reverse=True)
                    if so_files:
                        library = str(so_files[0])
                        break

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

    csrc = Path("oneDNN_extension_demo") / "csrc"
    kernels = csrc / "kernels"
    cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
    link_args = [str(library_path)]
    if os.name != "nt":
        link_args.append(f"-Wl,-rpath,{library_path.parent}")

    op_context_ext = CppExtension(
        name="oneDNN_extension_demo._C",
        sources=[str(csrc / "op_context.cpp")],
        include_dirs=[str(include_path)],
        define_macros=[("ODNN_DEMO_USE_DNNL", "1")],
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=link_args,
    )

    hijack_cxx_flags = list(cxx_flags)
    if os.name != "nt":
        hijack_cxx_flags.append("-std=c++17")
    hijack_ext = CppExtension(
        name="oneDNN_extension_demo._hijack",
        sources=[
            str(csrc / "onednn_hijack.cpp"),
            str(kernels / "eltwise.cpp"),
            str(kernels / "pooling.cpp"),
            str(kernels / "softmax.cpp"),
            str(kernels / "binary.cpp"),
            str(kernels / "unary.cpp"),
            str(kernels / "normalization.cpp"),
            str(kernels / "matmul.cpp"),
        ],
        include_dirs=[str(include_path)],
        define_macros=[("ODNN_DEMO_USE_DNNL", "1")],
        extra_compile_args={"cxx": hijack_cxx_flags},
        extra_link_args=link_args,
    )

    version = torch.__version__.split("+", 1)[0].split(".")
    torch_requirement = (
        f"torch>={version[0]}.{version[1]},"
        f"<{version[0]}.{int(version[1]) + 1}"
    )
    return (
        [op_context_ext, hijack_ext],
        {"build_ext": BuildExtension.with_options(use_ninja=True)},
        torch_requirement,
    )


if BUILD_NATIVE:
    ext_modules, cmdclass, torch_requirement = _native_build_config()
else:
    ext_modules, cmdclass, torch_requirement = [], {}, "torch>=2.8,<2.9"


setup(
    version=PACKAGE_VERSION,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[torch_requirement, "typing-extensions>=4.0"],
)
