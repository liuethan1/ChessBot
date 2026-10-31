"""Reading EPD files: perft suites (``;D1 20 ;D2 400``) and test suites (``bm``/``am`` opcodes)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess


@dataclass(frozen=True)
class PerftEntry:
    fen: str
    counts: dict[int, int]


@dataclass(frozen=True)
class SuiteEntry:
    id: str
    board: chess.Board
    best: tuple[chess.Move, ...]
    avoid: tuple[chess.Move, ...]
    points: dict[chess.Move, int]
    """STS-style partial credit (``c8``/``c9``); empty for plain bm/am suites."""


def _lines(path: Path) -> Iterator[str]:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            yield line


def read_perft_suite(path: Path) -> list[PerftEntry]:
    entries = []
    for line in _lines(path):
        fen, *fields = (part.strip() for part in line.split(";"))
        counts = {}
        for field in fields:
            depth, nodes = field.split()
            counts[int(depth.removeprefix("D"))] = int(nodes)
        board = chess.Board(fen)
        entries.append(PerftEntry(board.fen(), counts))
    return entries


def read_test_suite(path: Path) -> list[SuiteEntry]:
    entries = []
    for index, line in enumerate(_lines(path), start=1):
        board, ops = chess.Board.from_epd(line)
        best = tuple(_moves(ops.get("bm")))
        avoid = tuple(_moves(ops.get("am")))
        if not best and not avoid:
            raise ValueError(f"{path}:{index}: no bm or am opcode")
        points: dict[chess.Move, int] = {}
        if isinstance(ops.get("c9"), str) and isinstance(ops.get("c8"), str):
            ucis = str(ops["c9"]).split()
            scores = str(ops["c8"]).split()
            points = {chess.Move.from_uci(u): int(s) for u, s in zip(ucis, scores, strict=True)}
        entry_id = str(ops.get("id", f"{path.stem}.{index:03d}"))
        entries.append(SuiteEntry(entry_id, board, best, avoid, points))
    return entries


def _moves(value: object) -> list[chess.Move]:
    if value is None:
        return []
    if isinstance(value, list):
        return [m for m in value if isinstance(m, chess.Move)]
    raise ValueError(f"expected a move list, got {value!r}")
