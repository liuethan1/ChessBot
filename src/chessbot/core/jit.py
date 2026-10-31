"""numba configuration shared by every jitted function in ``chessbot.core``.

All jitted functions are compiled with ``cache=True`` so that only the very first run on a machine
pays the compile cost. numba's own cache only checks the timestamp of the file that defines each
function, so editing a callee in another module can leave a stale caller in the cache. To avoid
that, the cache lives in a directory keyed by a hash of every source file in the jitted packages
(``chessbot.core`` and ``chessbot.search``, plus the Python and numba versions); any edit to them
moves to a fresh cache directory.

Set ``CHESSBOT_CACHE_DIR`` to choose the cache root, or ``NUMBA_CACHE_DIR`` to bypass this logic.
This module must be imported before any jitted function is defined, because numba reads
``numba.config.CACHE_DIR`` when a function is decorated.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import numba

F = TypeVar("F", bound=Callable[..., Any])

CORE_DIR = Path(__file__).resolve().parent
JIT_DIRS = (CORE_DIR, CORE_DIR.parent / "search")


def source_digest() -> str:
    h = hashlib.sha256()
    h.update(f"{sys.version}|{numba.__version__}".encode())
    for directory in JIT_DIRS:
        for path in sorted(directory.glob("*.py")):
            h.update(f"{directory.name}/{path.name}".encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def cache_root() -> Path:
    env = os.environ.get("CHESSBOT_CACHE_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "chessbot"


def cache_dir() -> Path:
    if os.environ.get("NUMBA_CACHE_DIR"):
        return Path(os.environ["NUMBA_CACHE_DIR"])
    return cache_root() / f"numba-{source_digest()}"


if not os.environ.get("NUMBA_CACHE_DIR"):
    numba.config.CACHE_DIR = str(cache_dir())  # type: ignore[attr-defined]


def njit(func: F) -> F:
    """``numba.njit`` with the engine's settings (on-disk cache, GIL released)."""
    return cast(F, numba.njit(cache=True, nogil=True)(func))
