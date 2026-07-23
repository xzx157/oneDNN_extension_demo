"""Public entry points for the oneDNN extension demo."""

import os

from . import ops
from .frontend import optimize


def enable_aten_hijack():
    """Opt in to the experimental global aten CPU hijack extension."""

    from .cpp_extension import load_hijack_extension

    load_hijack_extension()


if os.environ.get("ODNN_ENABLE_ATEN_HIJACK") == "1":
    enable_aten_hijack()


__all__ = ["enable_aten_hijack", "optimize", "ops"]
