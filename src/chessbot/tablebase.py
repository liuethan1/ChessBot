"""Syzygy DTZ probing at the root, via python-chess.

WDL/DTZ files are opened from ``SyzygyPath``. The jitted search never calls this module: a 5-piece
probe is cheap at the root and too slow to run from every node in Python. In-search WDL is omitted
on purpose (it would need a numba reader).
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import chess.syzygy

from chessbot.core.board import Board
from chessbot.tools.common import EXTERNAL_DIR

DEFAULT_PROBE_LIMIT = 5
WDL_SCORE = 20000
"""Centipawn-scale score used for a root WDL of ±2 (uncursed win/loss)."""


def piece_count(fen: str) -> int:
    return sum(c.isalpha() for c in fen.split()[0])


def default_syzygy_path() -> str:
    env = os.environ.get("CHESSBOT_SYZYGY")
    if env:
        return env
    candidate = EXTERNAL_DIR / "syzygy"
    if candidate.is_dir() and any(candidate.glob("*.rtbw")):
        return str(candidate)
    return ""


def wdl_to_score(wdl: int) -> int:
    return wdl * (WDL_SCORE // 2)


class Tablebases:
    """Lazy Syzygy collection. ``set_path`` reloads when the UCI option changes."""

    def __init__(self) -> None:
        self.path: str = ""
        self._tb: chess.syzygy.Tablebase | None = None
        self._files = 0

    @property
    def loaded(self) -> bool:
        return self._tb is not None

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        self.path = ""
        self._files = 0

    def set_path(self, path: str) -> int:
        """Load every ``.rtbw``/``.rtbz`` in ``path`` (``os.pathsep``-separated). Returns files."""
        cleaned = path.strip()
        if cleaned == self.path and (self._tb is not None or not cleaned):
            return self._files
        self.close()
        if not cleaned:
            return 0
        tb = chess.syzygy.Tablebase()
        files = 0
        for part in cleaned.split(os.pathsep):
            directory = Path(part).expanduser()
            if directory.is_dir():
                files += tb.add_directory(str(directory))
        if files == 0:
            tb.close()
            return 0
        self._tb = tb
        self.path = cleaned
        self._files = files
        return files

    def probe_dtz_move(
        self, board: Board, *, probe_limit: int = DEFAULT_PROBE_LIMIT
    ) -> tuple[int, int] | None:
        """Best DTZ move and a WDL-scaled score, or ``None`` if this root is not in the tables."""
        if self._tb is None or probe_limit <= 0:
            return None
        fen = board.fen()
        if piece_count(fen) > probe_limit:
            return None
        root = chess.Board(fen)
        best: tuple[int, int, chess.Move] | None = None
        for move in root.legal_moves:
            root.push(move)
            try:
                if root.is_checkmate():
                    root.pop()
                    engine_move = board.parse_uci(move.uci())
                    return engine_move, WDL_SCORE
                wdl = self._tb.probe_wdl(root)
                dtz = self._tb.probe_dtz(root)
            except (KeyError, chess.syzygy.MissingTableError, ValueError):
                root.pop()
                continue
            root.pop()
            key = (-wdl, dtz)
            if best is None or key > best[:2]:
                best = (key[0], key[1], move)
        if best is None:
            return None
        try:
            engine_move = board.parse_uci(best[2].uci())
        except ValueError:
            return None
        return engine_move, wdl_to_score(best[0])
