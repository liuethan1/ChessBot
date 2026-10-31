"""Perft harness: checks a move generator's leaf counts against a reference suite.

Backends:

* ``reference``: python-chess (slow, but trusted).
* ``uci:COMMAND``: any UCI engine that implements ``go perft N`` (Stockfish's output format).
* ``py:MODULE:FUNCTION``: an in-process ``FUNCTION(fen: str, depth: int) -> int``.

When a count is wrong, the harness walks down the tree comparing per-move counts with the
reference until it finds the position whose move list differs.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess

from chessbot.tools.common import data_path, split_command
from chessbot.tools.epd import PerftEntry, read_perft_suite

SUITES = {"cpw": "cpw.epd", "standard": "standard.epd"}


class PerftBackend(Protocol):
    name: str

    def perft(self, fen: str, depth: int) -> int: ...

    def divide(self, fen: str, depth: int) -> dict[str, int]: ...

    def close(self) -> None: ...


def reference_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    moves = list(board.legal_moves)
    if depth == 1:
        return len(moves)
    total = 0
    for move in moves:
        board.push(move)
        total += reference_perft(board, depth - 1)
        board.pop()
    return total


def _divide_with(perft: Callable[[str, int], int], fen: str, depth: int) -> dict[str, int]:
    board = chess.Board(fen)
    result = {}
    for move in board.legal_moves:
        board.push(move)
        result[move.uci()] = perft(board.fen(), depth - 1)
        board.pop()
    return result


class ReferenceBackend:
    name = "reference"

    def perft(self, fen: str, depth: int) -> int:
        return reference_perft(chess.Board(fen), depth)

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        return _divide_with(self.perft, fen, depth)

    def close(self) -> None:
        pass


class FunctionBackend:
    def __init__(self, spec: str) -> None:
        module_name, _, func_name = spec.rpartition(":")
        if not module_name:
            raise ValueError(f"expected MODULE:FUNCTION, got {spec!r}")
        self.name = f"py:{spec}"
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        self._func: Callable[[str, int], int] = getattr(
            importlib.import_module(module_name), func_name
        )

    def perft(self, fen: str, depth: int) -> int:
        return int(self._func(fen, depth))

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        return _divide_with(self.perft, fen, depth)

    def close(self) -> None:
        pass


class UciBackend:
    def __init__(self, command: str) -> None:
        self.name = f"uci:{command}"
        self._proc = subprocess.Popen(
            split_command(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until(lambda line: line == "uciok")

    def _send(self, line: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_until(self, done: Callable[[str], bool]) -> list[str]:
        assert self._proc.stdout is not None
        lines = []
        for raw in self._proc.stdout:
            line = raw.strip()
            lines.append(line)
            if done(line):
                return lines
        raise RuntimeError(f"{self.name} exited before finishing")

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        self._send(f"position fen {fen}")
        self._send(f"go perft {depth}")
        lines = self._read_until(lambda line: line.startswith("Nodes searched"))
        result = {}
        for line in lines:
            move, sep, count = line.partition(":")
            if sep and not line.startswith("Nodes searched") and count.strip().isdigit():
                result[move.strip()] = int(count)
        return result

    def perft(self, fen: str, depth: int) -> int:
        self._send(f"position fen {fen}")
        self._send(f"go perft {depth}")
        lines = self._read_until(lambda line: line.startswith("Nodes searched"))
        return int(lines[-1].split(":")[1])

    def close(self) -> None:
        if self._proc.poll() is None:
            self._send("quit")
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def make_backend(spec: str) -> PerftBackend:
    if spec == "reference":
        return ReferenceBackend()
    if spec.startswith("uci:"):
        return UciBackend(spec.removeprefix("uci:"))
    if spec.startswith("py:"):
        return FunctionBackend(spec.removeprefix("py:"))
    raise ValueError(f"unknown backend {spec!r} (use reference, uci:COMMAND or py:MODULE:FUNC)")


@dataclass
class Mismatch:
    """Where a backend first disagrees with the reference move list."""

    fen: str
    path: list[str]
    depth: int
    total: tuple[int, int]
    """(got, expected) perft count at this position and depth."""
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    wrong_counts: dict[str, tuple[int, int]] = field(default_factory=dict)

    def describe(self) -> str:
        got, want = self.total
        parts = [f"after {' '.join(self.path) or '(root)'}: {self.fen}"]
        parts.append(f"  perft({self.depth}): got {got}, expected {want}")
        if not (self.missing or self.extra or self.wrong_counts):
            legal = " ".join(sorted(m.uci() for m in chess.Board(self.fen).legal_moves))
            parts.append(f"  move generation is wrong here; legal moves are: {legal}")
        if self.missing:
            parts.append(f"  missing moves: {' '.join(sorted(self.missing))}")
        if self.extra:
            parts.append(f"  illegal extra moves: {' '.join(sorted(self.extra))}")
        for move, (got, want) in sorted(self.wrong_counts.items()):
            parts.append(f"  {move}: got {got}, expected {want}")
        return "\n".join(parts)


def find_mismatch(backend: PerftBackend, fen: str, depth: int) -> Mismatch | None:
    """Descend from ``fen`` to the shallowest position whose own move generation is wrong."""
    reference = ReferenceBackend()
    path: list[str] = []
    while depth >= 1:
        total = (backend.perft(fen, depth), reference.perft(fen, depth))
        if total[0] == total[1]:
            return None
        got = backend.divide(fen, depth)
        want = reference.divide(fen, depth)
        missing = [m for m in want if m not in got]
        extra = [m for m in got if m not in want]
        wrong = {m: (got[m], want[m]) for m in want if m in got and got[m] != want[m]}
        # A py: backend's divide enumerates the reference move list, so a bug in this node's own
        # move generation shows up only as a total that disagrees with the per-move counts.
        if missing or extra or not wrong or depth == 1:
            return Mismatch(fen, path, depth, total, missing, extra, wrong)
        move = next(iter(wrong))
        board = chess.Board(fen)
        board.push_uci(move)
        fen, depth = board.fen(), depth - 1
        path.append(move)
    return None


@dataclass
class PerftResult:
    fen: str
    depth: int
    expected: int
    nodes: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.nodes == self.expected

    @property
    def nps(self) -> float:
        return self.nodes / self.seconds if self.seconds > 0 else 0.0


def select_depths(entry: PerftEntry, max_depth: int, max_nodes: int) -> list[int]:
    return [d for d, n in sorted(entry.counts.items()) if d <= max_depth and n <= max_nodes]


def run_suite(
    backend: PerftBackend,
    entries: list[PerftEntry],
    max_depth: int,
    max_nodes: int,
    deepest_only: bool = True,
) -> list[PerftResult]:
    results = []
    for entry in entries:
        depths = select_depths(entry, max_depth, max_nodes)
        if deepest_only:
            depths = depths[-1:]
        for depth in depths:
            start = time.perf_counter()
            nodes = backend.perft(entry.fen, depth)
            elapsed = time.perf_counter() - start
            results.append(PerftResult(entry.fen, depth, entry.counts[depth], nodes, elapsed))
    return results


def resolve_suite(name: str) -> Path:
    return data_path("perft", SUITES[name]) if name in SUITES else Path(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="reference", help="reference | uci:CMD | py:MOD:FUNC")
    parser.add_argument("--suite", default="cpw", help="cpw | standard | path to an EPD file")
    parser.add_argument("--fen", help="run a single position instead of a suite")
    parser.add_argument("--depth", type=int, default=4, help="depth for --fen / --divide")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=10_000_000, help="skip larger entries")
    parser.add_argument(
        "--all-depths", action="store_true", help="check every depth, not just the deepest"
    )
    parser.add_argument("--divide", action="store_true", help="print per-move counts for --fen")
    args = parser.parse_args(argv)

    backend = make_backend(args.backend)
    try:
        if args.fen:
            if args.divide:
                counts = backend.divide(args.fen, args.depth)
                for move, count in sorted(counts.items()):
                    print(f"{move}: {count}")
                print(f"\nNodes searched: {sum(counts.values())}")
                return 0
            expected = ReferenceBackend().perft(args.fen, args.depth)
            entries = [PerftEntry(args.fen, {args.depth: expected})]
        else:
            entries = read_perft_suite(resolve_suite(args.suite))

        results = run_suite(
            backend, entries, args.max_depth, args.max_nodes, deepest_only=not args.all_depths
        )
        failures = [r for r in results if not r.ok]
        for r in results:
            status = "ok  " if r.ok else "FAIL"
            print(
                f"{status} d{r.depth} {r.nodes:>12} / {r.expected:<12} "
                f"{r.seconds:7.2f}s {r.nps / 1e3:9.1f} knps  {r.fen}"
            )
        total_nodes = sum(r.nodes for r in results)
        total_time = sum(r.seconds for r in results)
        nps = total_nodes / total_time if total_time else 0.0
        print(
            f"\n{backend.name}: {len(results) - len(failures)}/{len(results)} passed, "
            f"{total_nodes} nodes in {total_time:.2f}s ({nps / 1e3:.1f} knps)"
        )
        for r in failures:
            mismatch = find_mismatch(backend, r.fen, r.depth)
            if mismatch:
                print("\nFirst difference:\n" + mismatch.describe())
        return 1 if failures else 0
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
