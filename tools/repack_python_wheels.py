import argparse
import base64
import copy
import csv
import hashlib
import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from packaging.version import InvalidVersion, Version


PACKAGE_NAME = "oneDNN_extension_demo"
NATIVE_BUILD_INPUTS = {
    "LICENSE",
    "MANIFEST.in",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "setup.py",
    "tools/build_onednn.sh",
}
RECORD_SIGNATURES = {"RECORD", "RECORD.jws", "RECORD.p7s"}


def _is_python_payload(name):
    path = PurePosixPath(name)
    return (
        path.parts
        and path.parts[0] == PACKAGE_NAME
        and (path.suffix in {".py", ".pyi"} or path.name == "py.typed")
    )


def _requires_native_build(name):
    path = PurePosixPath(name)
    if name in NATIVE_BUILD_INPUTS or (path.parts and path.parts[0] == "LICENSES"):
        return True
    return (
        path.parts
        and path.parts[0] == PACKAGE_NAME
        and not _is_python_payload(name)
    )


def _check_python_only_changes(base_ref):
    result = subprocess.run(
        ["git", "diff", "--name-only", "-z", base_ref, "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
    )
    changed = [
        name.decode("utf-8")
        for name in result.stdout.split(b"\0")
        if name
    ]
    unsafe = [name for name in changed if _requires_native_build(name)]
    if unsafe:
        details = "\n  - ".join(unsafe)
        raise RuntimeError(
            "Fast repacking cannot be used because native or packaging inputs "
            f"changed since {base_ref}:\n  - {details}\n"
            "Run Build native wheels instead."
        )
    print(f"Python-only safety check passed against {base_ref}.")


def _wheel_version(value):
    try:
        normalized = str(Version(value))
    except InvalidVersion as error:
        raise ValueError(f"invalid package version: {value}") from error
    escaped = re.sub(r"[^\w\d.]+", "_", normalized, flags=re.UNICODE)
    return normalized, escaped


def _metadata_with_version(data, version):
    pattern = re.compile(br"^Version:[ \t]*[^\r\n]*", re.MULTILINE)
    updated, substitutions = pattern.subn(
        f"Version: {version}".encode("ascii"), data, count=1
    )
    if substitutions != 1:
        raise RuntimeError("wheel METADATA does not contain a Version header")
    return updated


def _metadata_version(data):
    match = re.search(br"^Version:[ \t]*([^\r\n]+)", data, re.MULTILINE)
    if match is None:
        raise RuntimeError("wheel METADATA does not contain a Version header")
    return match.group(1).decode("ascii").strip()


def _python_sources(package_dir):
    sources = {}
    for path in sorted(package_dir.rglob("*")):
        relative = path.relative_to(package_dir)
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        archive_name = (PurePosixPath(PACKAGE_NAME) / relative.as_posix()).as_posix()
        if not _is_python_payload(archive_name):
            continue
        data = path.read_bytes()
        if path.suffix == ".py":
            compile(data, str(path), "exec")
        sources[archive_name] = data
    if f"{PACKAGE_NAME}/__init__.py" not in sources:
        raise RuntimeError(f"{package_dir} does not contain __init__.py")
    return sources


def _zip_info(name, source=None):
    if source is not None:
        info = copy.copy(source)
        info.filename = name
        return info
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _record_data(entries, record_name):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name in sorted(entries):
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(entries[name][0]).digest()
        ).rstrip(b"=").decode("ascii")
        writer.writerow((name, f"sha256={digest}", len(entries[name][0])))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode("utf-8")


def _verify_record(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise RuntimeError(f"{wheel.name} has {len(record_names)} RECORD files")
        record_name = record_names[0]
        rows = csv.reader(io.TextIOWrapper(archive.open(record_name), encoding="utf-8"))
        records = {row[0]: row[1:] for row in rows}
        if set(records) != set(names):
            raise RuntimeError(f"{wheel.name} RECORD does not match archive contents")
        for name in names:
            if name == record_name:
                continue
            digest, size = records[name]
            data = archive.read(name)
            expected = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode("ascii")
            if digest != f"sha256={expected}" or size != str(len(data)):
                raise RuntimeError(f"{wheel.name} has an invalid RECORD entry: {name}")


def _repack(source_wheel, output_dir, version, escaped_version, sources):
    with zipfile.ZipFile(source_wheel) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        dist_info_dirs = {
            info.filename.split("/", 1)[0]
            for info in infos
            if ".dist-info/" in info.filename
        }
        if len(dist_info_dirs) != 1:
            raise RuntimeError(
                f"{source_wheel.name} has {len(dist_info_dirs)} dist-info directories"
            )
        old_dist_info = dist_info_dirs.pop()
        metadata_name = f"{old_dist_info}/METADATA"
        old_version = _metadata_version(archive.read(metadata_name))
        if Version(old_version) == Version(version):
            raise RuntimeError(
                f"new version {version} matches the source wheel version; "
                "use a new version for a changed package"
            )
        _, escaped_old_version = _wheel_version(old_version)
        suffix = f"-{escaped_old_version}.dist-info"
        if not old_dist_info.endswith(suffix):
            raise RuntimeError(
                f"cannot derive distribution name from {old_dist_info}"
            )
        distribution = old_dist_info[: -len(suffix)]
        new_dist_info = f"{distribution}-{escaped_version}.dist-info"

        old_filename_prefix = f"{distribution}-{escaped_old_version}-"
        if not source_wheel.name.startswith(old_filename_prefix):
            raise RuntimeError(
                f"wheel filename does not match its metadata: {source_wheel.name}"
            )
        output_name = (
            f"{distribution}-{escaped_version}-"
            f"{source_wheel.name[len(old_filename_prefix):]}"
        )
        output_wheel = output_dir / output_name
        if output_wheel.exists():
            raise RuntimeError(f"refusing to overwrite {output_wheel}")

        entries = {}
        for info in infos:
            name = info.filename
            if _is_python_payload(name):
                continue
            if name.startswith(f"{old_dist_info}/"):
                leaf = name.rsplit("/", 1)[-1]
                if leaf in RECORD_SIGNATURES:
                    continue
                new_name = f"{new_dist_info}/{name.split('/', 1)[1]}"
            else:
                new_name = name
            data = archive.read(info)
            if new_name == f"{new_dist_info}/METADATA":
                data = _metadata_with_version(data, version)
            entries[new_name] = (data, _zip_info(new_name, info))

    for name, data in sources.items():
        entries[name] = (data, _zip_info(name))

    record_name = f"{new_dist_info}/RECORD"
    record = _record_data(entries, record_name)
    entries[record_name] = (record, _zip_info(record_name))

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=f".{output_name}.", suffix=".tmp", delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for name in sorted(entries):
                data, info = entries[name]
                archive.writestr(info, data)
        temporary_path.replace(output_wheel)
    finally:
        temporary_path.unlink(missing_ok=True)

    _verify_record(output_wheel)
    print(f"Repacked {source_wheel.name} -> {output_wheel.name}")
    return output_wheel


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replace Python sources in repaired wheels without rebuilding native code."
    )
    parser.add_argument("wheels", nargs="*", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-ref")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--require-arch", action="append", default=[])
    return parser.parse_args()


def main():
    args = _parse_args()
    version, escaped_version = _wheel_version(args.version)
    if args.base_ref:
        _check_python_only_changes(args.base_ref)
    if args.check_only:
        if not args.base_ref:
            raise ValueError("--check-only requires --base-ref")
        return
    if not args.wheels:
        raise ValueError("at least one input wheel is required")
    package_dir = Path(__file__).resolve().parents[1] / PACKAGE_NAME
    sources = _python_sources(package_dir)
    wheels = [wheel.resolve() for wheel in args.wheels]
    missing = [wheel for wheel in wheels if not wheel.is_file()]
    if missing:
        raise FileNotFoundError(f"wheel files not found: {missing}")
    for architecture in args.require_arch:
        if not any(wheel.name.endswith(f"_{architecture}.whl") for wheel in wheels):
            raise RuntimeError(f"input artifacts do not contain a {architecture} wheel")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _repack(wheel, args.output_dir, version, escaped_version, sources)
        for wheel in wheels
    ]
    print(f"Created {len(outputs)} Python-only wheel(s) at {args.output_dir}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
