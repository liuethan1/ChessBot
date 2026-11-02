"""Polyglot opening book, probed from Python at the root.

The jitted search never sees this module. python-chess is used only to hash the root FEN with
the Polyglot Zobrist keys and to decode a book move; that move is then matched against the
engine's own legal list.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.polyglot

from chessbot.core.board import Board
from chessbot.core.position import ST_FULLMOVE
from chessbot.tools.common import data_path

DEFAULT_BOOK_NAME = "book.bin"
DEFAULT_BOOK_DEPTH = 32
"""Use the book for this many plies from the start of the game (fullmove clock)."""


def default_book_path() -> Path:
    return data_path(DEFAULT_BOOK_NAME)


def game_ply(board: Board) -> int:
    """Plies already played in the game, from the FEN fullmove number and side to move."""
    fullmove = max(1, int(board.pos.st[ST_FULLMOVE]))
    return 2 * (fullmove - 1) + (0 if board.white_to_move else 1)


def encode_polyglot_move(board: chess.Board, move: chess.Move) -> int:
    """Raw 16-bit Polyglot move (castling encoded as king-captures-rook)."""
    from_sq, to_sq = move.from_square, move.to_square
    if board.is_castling(move):
        to_file = 7 if board.is_kingside_castling(move) else 0
        to_sq = chess.square(to_file, chess.square_rank(from_sq))
    promo = 0 if move.promotion is None else move.promotion - 1
    return to_sq | (from_sq << 6) | (promo << 12)


class OpeningBook:
    """Memory-mapped Polyglot reader. ``set_path`` reopens when the UCI option changes."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self._reader: chess.polyglot.MemoryMappedReader | None = None

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self.path = None

    def set_path(self, path: str | Path | None) -> None:
        if path is None or str(path).strip() == "":
            self.close()
            return
        resolved = Path(path).expanduser()
        if self.path == resolved and self._reader is not None:
            return
        self.close()
        if not resolved.is_file():
            return
        self._reader = chess.polyglot.open_reader(resolved)
        self.path = resolved

    def probe(self, board: Board, *, best: bool = True, minimum_weight: int = 1) -> int | None:
        """Return an engine move, or ``None`` if the position is not in the book."""
        if self._reader is None:
            return None
        root = chess.Board(board.fen())
        try:
            if best:
                entry = self._reader.find(root, minimum_weight=minimum_weight)
            else:
                entry = self._reader.weighted_choice(root)
        except IndexError:
            return None
        try:
            return board.parse_uci(entry.move.uci())
        except ValueError:
            return None
