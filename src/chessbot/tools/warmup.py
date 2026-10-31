"""Warm-start check: how long a fresh process needs before the jitted engine core is ready.

Without ``--check`` this warms up in-process and prints the time. With ``--check SECONDS`` it
starts two fresh interpreters: the first fills the numba cache if it is empty (a cold compile),
the second must import and warm up within ``SECONDS`` or the command exits with status 1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

CHILD = "from chessbot.core import warmup; warmup()"


def fresh_process_seconds() -> float:
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", CHILD], check=True)
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", type=float, metavar="SECONDS", help="fail if a warm start takes longer"
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    from chessbot.core import warmup
    from chessbot.core.jit import cache_dir

    imported = time.perf_counter() - start
    print(f"numba cache: {cache_dir()}")
    if args.check is None:
        print(f"import {imported:.2f}s, warm-up {warmup():.2f}s")
        return 0

    first = fresh_process_seconds()
    print(f"first process (compiles if the cache was empty): {first:.2f}s")
    warm = fresh_process_seconds()
    ok = warm <= args.check
    print(f"warm start: {warm:.2f}s (limit {args.check:.2f}s) {'ok' if ok else 'TOO SLOW'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
