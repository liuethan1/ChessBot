"""Transposition table: one entry per slot, indexed by the low bits of the Zobrist key.

The table is a named tuple of parallel numpy arrays so it can be passed into jitted code. Each
entry takes 16 bytes. Mate scores are stored relative to the node (distance from the node, not
from the root) and converted back on probe, so an entry stays valid wherever it is reached.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from chessbot.core.jit import njit

BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT = 0, 1, 2, 3
ENTRY_BYTES = 16
DEFAULT_HASH_MB = 16
MAX_HASH_MB = 4096

TT_MATE_BOUND = 32000 - 128
"""Same value as ``negamax.MATE_BOUND`` (``negamax`` imports this module, not the other way)."""


class TransTable(NamedTuple):
    keys: npt.NDArray[np.uint64]
    moves: npt.NDArray[np.uint16]
    scores: npt.NDArray[np.int16]
    evals: npt.NDArray[np.int16]
    depths: npt.NDArray[np.int8]
    bounds: npt.NDArray[np.uint8]
    gens: npt.NDArray[np.uint8]
    gen: npt.NDArray[np.int64]
    """(1,): generation of the current search, used to prefer replacing stale entries."""


def new_tt(mb: int = DEFAULT_HASH_MB) -> TransTable:
    """A table of the largest power-of-two entry count that fits in ``mb`` megabytes."""
    want = max(1, min(mb, MAX_HASH_MB)) * (1 << 20) // ENTRY_BYTES
    n = 1 << (want.bit_length() - 1)
    return TransTable(
        np.zeros(n, np.uint64),
        np.zeros(n, np.uint16),
        np.zeros(n, np.int16),
        np.zeros(n, np.int16),
        np.zeros(n, np.int8),
        np.zeros(n, np.uint8),
        np.zeros(n, np.uint8),
        np.zeros(1, np.int64),
    )


def clear_tt(tt: TransTable) -> None:
    for a in (tt.keys, tt.moves, tt.scores, tt.evals, tt.depths, tt.bounds, tt.gens, tt.gen):
        a.fill(0)


def new_search_generation(tt: TransTable) -> None:
    tt.gen[0] = (int(tt.gen[0]) + 1) & 255


@njit
def tt_index(tt: TransTable, key: np.uint64) -> int:
    return int(key & np.uint64(tt.keys.shape[0] - 1))


@njit
def score_to_tt(score: int, ply: int) -> int:
    if score > TT_MATE_BOUND:
        return score + ply
    if score < -TT_MATE_BOUND:
        return score - ply
    return score


@njit
def score_from_tt(score: int, ply: int) -> int:
    if score > TT_MATE_BOUND:
        return score - ply
    if score < -TT_MATE_BOUND:
        return score + ply
    return score


@njit
def tt_store(
    tt: TransTable,
    key: np.uint64,
    move: int,
    score: int,
    ev: int,
    depth: int,
    bound: int,
    ply: int,
) -> None:
    i = tt_index(tt, key)
    gen = tt.gen[0]
    same = tt.keys[i] == key
    if same and move == 0:
        move = int(tt.moves[i])
    if not same or bound == BOUND_EXACT or depth + 3 >= tt.depths[i] or tt.gens[i] != gen:
        tt.keys[i] = key
        tt.moves[i] = move
        tt.scores[i] = score_to_tt(score, ply)
        tt.evals[i] = ev
        tt.depths[i] = max(-128, min(127, depth))
        tt.bounds[i] = bound
        tt.gens[i] = gen
