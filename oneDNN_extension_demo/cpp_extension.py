import os
import platform
import shutil
from pathlib import Path

import torch


_LOADED = False


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
    global _LOADED
    if _LOADED:
        return

    _check_build_tools()
    _check_windows_architecture()
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "op_context.cpp"
    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    architecture = platform.machine().lower() or "unknown"
    extension_name = f"odnn_demo_op_context_{architecture}_v2"
    try:
        load(
            name=extension_name,
            sources=[str(source)],
            extra_cflags=extra_cflags,
            is_python_module=False,
            verbose=os.environ.get("ODNN_DEMO_CPP_VERBOSE") == "1",
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to build the demo C++ OpContext extension. "
            "Set ODNN_DEMO_CPP_VERBOSE=1 for the full compiler command. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    if not hasattr(torch.classes, "odnn_prepack"):
        raise RuntimeError("C++ OpContext classes were not registered.")
    _LOADED = True
