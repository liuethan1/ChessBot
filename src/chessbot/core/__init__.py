"""Engine core: numba-jitted bitboards, move generation, and make/unmake."""

from __future__ import annotations

import time

from chessbot.core.board import Board
from chessbot.core.perft import perft_fen
from chessbot.core.position import START_FEN

WARMUP_FENS = (
    START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
)


def warmup() -> float:
    """Compile (or load from the on-disk cache) every jitted entry point; returns seconds taken.

    Call this before the clock starts, for example while handling ``uci``/``isready``.
    """
    from chessbot.core.position import (
        compute_key,
        make_null_move,
        unmake_null_move,
    )

    start = time.perf_counter()
    for fen in WARMUP_FENS:
        board = Board(fen)
        board.perft(2)
        board.legal_moves(tactical_only=True)
        board.in_check()
        board.push(board.legal_moves()[0])
        board.pop()
        make_null_move(board.pos, board.t)
        unmake_null_move(board.pos, board.t)
        compute_key(board.pos, board.t)
    return time.perf_counter() - start


__all__ = ["Board", "perft_fen", "warmup"]
