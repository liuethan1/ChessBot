"""Helpers shared by the measurement tools."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from importlib import resources
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = Path(os.environ.get("CHESSBOT_EXTERNAL", REPO_ROOT / "external"))


def data_path(*parts: str) -> Path:
    """Path to a file shipped in ``chessbot/data``."""
    return Path(str(resources.files("chessbot").joinpath("data", *parts)))


def find_binary(name: str, env_var: str) -> str | None:
    """Locate a tool via ``$env_var``, then ``PATH``, then ``external/bin``, then ``/usr/games``."""
    candidates = [os.environ.get(env_var), shutil.which(name)]
    candidates += [str(EXTERNAL_DIR / "bin" / name), f"/usr/games/{name}"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def split_command(command: str) -> list[str]:
    """Split an engine command line such as ``"python -m chessbot"`` into argv.

    On Windows, POSIX shlex rules would treat the backslashes in paths as escapes, so the
    command is split in non-POSIX mode and quotes around whole arguments are removed.
    """
    if sys.platform == "win32":
        argv = [_unquote(arg) for arg in shlex.split(command, posix=False)]
    else:
        argv = shlex.split(command)
    if not argv:
        raise ValueError("empty engine command")
    return argv


def _unquote(arg: str) -> str:
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "'\"":
        return arg[1:-1]
    return arg


def parse_options(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``NAME=VALUE`` UCI option arguments."""
    options: dict[str, str] = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"expected NAME=VALUE, got {pair!r}")
        options[name.strip()] = value.strip()
    return options
