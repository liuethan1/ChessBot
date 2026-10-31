"""Python-facing wrapper around the jitted position (for UCI handling, tools, and tests)."""

from __future__ import annotations

import numpy as np

from chessbot.core.bitboard import WHITE, Tables, tables
from chessbot.core.movegen import MAX_MOVES, generate_moves, in_check
from chessbot.core.perft import perft
from chessbot.core.position import (
    MAX_PLY,
    ST_PLY,
    ST_SIDE,
    START_FEN,
    Position,
    from_fen,
    make_move,
    move_to_uci,
    to_fen,
    unmake_move,
)


class Board:
    def __init__(self, fen: str = START_FEN, t: Tables | None = None) -> None:
        self.t = t if t is not None else tables()
        self.pos: Position = from_fen(fen, self.t)
        self._buf = np.empty(MAX_MOVES, np.int32)

    def fen(self) -> str:
        return to_fen(self.pos)

    @property
    def ply(self) -> int:
        return int(self.pos.st[ST_PLY])

    @property
    def white_to_move(self) -> bool:
        return int(self.pos.st[ST_SIDE]) == WHITE

    @property
    def key(self) -> int:
        return int(self.pos.keys[self.ply])

    def legal_moves(self, tactical_only: bool = False) -> list[int]:
        n = generate_moves(self.pos, self.t, self._buf, tactical_only)
        return [int(m) for m in self._buf[:n]]

    def legal_uci(self) -> list[str]:
        return [move_to_uci(m) for m in self.legal_moves()]

    def in_check(self) -> bool:
        return bool(in_check(self.pos, self.t))

    def parse_uci(self, uci: str) -> int:
        for m in self.legal_moves():
            if move_to_uci(m) == uci:
                return m
        raise ValueError(f"illegal move {uci!r} in {self.fen()}")

    def push(self, move: int) -> None:
        if self.ply >= MAX_PLY:
            raise OverflowError(f"more than {MAX_PLY} plies")
        make_move(self.pos, self.t, move)

    def push_uci(self, uci: str) -> int:
        move = self.parse_uci(uci)
        self.push(move)
        return move

    def pop(self) -> None:
        if self.ply == 0:
            raise IndexError("no move to undo")
        unmake_move(self.pos, self.t)

    def perft(self, depth: int) -> int:
        return int(perft(self.pos, self.t, depth))

    def divide(self, depth: int) -> dict[str, int]:
        result = {}
        for m in self.legal_moves():
            self.push(m)
            result[move_to_uci(m)] = self.perft(depth - 1)
            self.pop()
        return result
