"""Perft (leaf-node counting) with bulk counting at the last ply."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import Tables, tables
from chessbot.core.jit import njit
from chessbot.core.movegen import MAX_MOVES, generate_moves
from chessbot.core.position import Position, from_fen, make_move, unmake_move


@njit
def perft(pos: Position, t: Tables, depth: int) -> int:
    """Iterative: numba segfaults when it reloads a cached self-recursive function."""
    if depth <= 0:
        return 1
    buf: npt.NDArray[np.int32] = np.empty((depth, MAX_MOVES), np.int32)
    count = np.zeros(depth, np.int64)
    index = np.zeros(depth, np.int64)
    count[0] = generate_moves(pos, t, buf[0], False)
    if depth == 1:
        return int(count[0])
    total = 0
    ply = 0
    while ply >= 0:
        if index[ply] == count[ply]:
            ply -= 1
            if ply >= 0:
                unmake_move(pos, t)
            continue
        make_move(pos, t, buf[ply, index[ply]])
        index[ply] += 1
        child = ply + 1
        if child == depth - 1:
            total += generate_moves(pos, t, buf[child], False)
            unmake_move(pos, t)
        else:
            count[child] = generate_moves(pos, t, buf[child], False)
            index[child] = 0
            ply = child
    return total


def perft_fen(fen: str, depth: int) -> int:
    """Entry point for ``chessbot-perft --backend py:chessbot.core.perft:perft_fen``."""
    t = tables()
    return int(perft(from_fen(fen, t), t, depth))
