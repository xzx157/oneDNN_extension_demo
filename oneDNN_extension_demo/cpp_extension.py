import os
import importlib.machinery
import platform
import shutil
from pathlib import Path

import torch


_LOADED = False
_LOAD_SOURCE = None


def _find_prebuilt_extension():
    package_dir = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = package_dir / f"_C{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _load_prebuilt_extension():
    candidate = _find_prebuilt_extension()
    if candidate is None:
        return False
    try:
        torch.ops.load_library(str(candidate))
    except Exception as error:
        raise RuntimeError(
            "The bundled native extension could not be loaded. Its wheel may "
            "not match the installed PyTorch or platform. Reinstall without "
            "--no-deps and use a wheel built for this PyTorch minor version. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error
    return True


def _validate_registration():
    try:
        namespace = torch.classes.odnn_prepack
        namespace.ConvolutionOpContext
        namespace.LinearOpContext
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "The native extension loaded but did not register the expected "
            "odnn_prepack classes."
        ) from error


def _find_dnnl_config():
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
                (str(path) for path in candidates if path.is_file()), None
            )

    if bool(include_dir) != bool(library):
        raise RuntimeError(
            "Native oneDNN requires both DNNL_INCLUDE_DIR and DNNL_LIBRARY "
            "(or a DNNL_ROOT containing include/ and lib/)."
        )
    if not include_dir:
        return [], [], False

    include_path = Path(include_dir).expanduser().resolve()
    library_path = Path(library).expanduser().resolve()
    header = include_path / "oneapi" / "dnnl" / "dnnl.hpp"
    if not header.is_file():
        raise RuntimeError(f"oneDNN header not found: {header}")
    if not library_path.is_file():
        raise RuntimeError(f"oneDNN library not found: {library_path}")

    ldflags = [str(library_path)]
    if os.name != "nt":
        ldflags.append(f"-Wl,-rpath,{library_path.parent}")
    return [str(include_path)], ldflags, True


def _check_build_tools():
    missing = []
    if os.name == "nt":
        for executable in ("cl", "link"):
            if shutil.which(executable) is None:
                missing.append(f"{executable}.exe")
    else:
        compiler = os.environ.get("CXX")
        compiler_found = compiler is not None and shutil.which(compiler) is not None
        if not compiler_found:
            compiler_found = any(
                shutil.which(executable) is not None
                for executable in ("c++", "g++", "clang++")
            )
        if not compiler_found:
            missing.append("CXX/c++/g++/clang++")

    if shutil.which("ninja") is None:
        missing.append("ninja")

    if missing:
        tools = ", ".join(missing)
        if os.name == "nt":
            hint = (
                "Install Visual Studio Build Tools with 'Desktop development "
                "with C++', install Ninja, then start Python/your IDE from an "
                "x64 Native Tools Command Prompt so cl.exe is on PATH."
            )
        else:
            hint = (
                "Install build-essential (or Clang) and Ninja, then retry. "
                "Set CXX when selecting a non-default compiler."
            )
        raise RuntimeError(f"Missing C++ build tools: {tools}. {hint}")


def _check_windows_architecture():
    if os.name != "nt":
        return

    python_arch = platform.architecture()[0]
    target_arch = os.environ.get("VSCMD_ARG_TGT_ARCH", "").lower()
    platform_target = os.environ.get("Platform", "").lower()
    if python_arch != "64bit":
        raise RuntimeError(
            f"The demo requires 64-bit Python, but found {python_arch}."
        )
    if target_arch in ("x86", "win32") or platform_target in ("x86", "win32"):
        raise RuntimeError(
            "MSVC is configured for an x86 target, but PyTorch is x64. "
            "Open 'x64 Native Tools Command Prompt for VS 2022' or run "
            "VsDevCmd.bat -arch=x64 -host_arch=x64 before starting Python."
        )


def load_cpp_extension():
    global _LOADED, _LOAD_SOURCE
    if _LOADED:
        return

    if _load_prebuilt_extension():
        _validate_registration()
        _LOADED = True
        _LOAD_SOURCE = "prebuilt-wheel"
        return

    _check_build_tools()
    _check_windows_architecture()
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "op_context.cpp"
    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    include_paths, extra_ldflags, use_native_dnnl = _find_dnnl_config()
    if use_native_dnnl:
        extra_cflags.append(
            "/DODNN_DEMO_USE_DNNL=1"
            if os.name == "nt"
            else "-DODNN_DEMO_USE_DNNL=1"
        )
    architecture = platform.machine().lower() or "unknown"
    backend = "dnnl" if use_native_dnnl else "aten"
    extension_name = f"odnn_demo_op_context_{architecture}_{backend}_v8"
    try:
        load(
            name=extension_name,
            sources=[str(source)],
            extra_cflags=extra_cflags,
            extra_include_paths=include_paths,
            extra_ldflags=extra_ldflags,
            is_python_module=False,
            verbose=os.environ.get("ODNN_DEMO_CPP_VERBOSE") == "1",
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to build the demo C++ OpContext extension. "
            "Set ODNN_DEMO_CPP_VERBOSE=1 for the full compiler command. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    _validate_registration()
    _LOADED = True
    _LOAD_SOURCE = "jit"


def cpp_extension_status():
    """Return whether the native extension is loaded and where it came from."""
    return {"loaded": _LOADED, "source": _LOAD_SOURCE}
