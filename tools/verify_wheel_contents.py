import sys
import zipfile
from pathlib import Path


def main():
    wheels = [Path(argument) for argument in sys.argv[1:]]
    if not wheels:
        raise SystemExit("usage: verify_wheel_contents.py WHEEL [WHEEL ...]")

    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
        native_extensions = [
            name
            for name in names
            if name.startswith("oneDNN_extension_demo/_C")
            and name.endswith(".so")
        ]
        shared_libraries = [
            name for name in names if ".so" in Path(name).name
        ]
        dnnl_libraries = [
            name
            for name in shared_libraries
            if Path(name).name.startswith("libdnnl")
        ]
        bundled_torch_libraries = [
            name
            for name in shared_libraries
            if Path(name).name.startswith(("libtorch", "libc10"))
        ]
        if not native_extensions:
            raise RuntimeError(f"{wheel.name} does not contain the native _C library")
        if not dnnl_libraries:
            raise RuntimeError(
                f"{wheel.name} does not contain a bundled oneDNN library; "
                f"shared libraries: {shared_libraries}"
            )
        if bundled_torch_libraries:
            raise RuntimeError(
                f"{wheel.name} unexpectedly bundles PyTorch libraries: "
                f"{bundled_torch_libraries}"
            )
        print(wheel.name, native_extensions, dnnl_libraries)


if __name__ == "__main__":
    main()
