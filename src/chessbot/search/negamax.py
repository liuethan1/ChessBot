"""Fail-soft negamax alpha-beta with quiescence, written as a resumable loop over an explicit stack.

numba segfaults when it reloads a cached self-recursive function, so the search keeps one frame
per ply in a :class:`SearchStack` instead of recursing. The same property makes it resumable:
:func:`run` searches at most ``budget`` nodes and returns, leaving the position and the stack mid-
tree, and the next call carries on. The Python driver uses this to read the clock and react to
``stop`` between chunks without the jitted code needing a clock of its own.

A frame whose remaining depth is 0 or less is a quiescence node: stand pat, then captures and
queen promotions only (or every evasion when in check).

Every completed frame is stored in the transposition table, and non-PV frames (null window) return
straight from a table entry that is deep enough and whose bound proves the result. The table move
is searched first.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import PAWN, QUEEN, Tables
from chessbot.core.jit import njit
from chessbot.core.movegen import MAX_MOVES, generate_moves, in_check
from chessbot.core.position import (
    CAPTURE,
    EP_CAPTURE,
    PROMO,
    ST_HALFMOVE,
    ST_PLY,
    Position,
    make_move,
    unmake_move,
)
from chessbot.search.evaluate import EvalTables, evaluate, insufficient_material
from chessbot.search.tt import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    TT_MATE_BOUND,
    TransTable,
    score_from_tt,
    tt_index,
    tt_store,
)

MAX_SEARCH_PLY = 128
MATE = 32000
INF = 32500
MATE_BOUND = MATE - MAX_SEARCH_PLY
"""Scores beyond +/- ``MATE_BOUND`` are mates; ``MATE - |score|`` is the distance in plies."""
assert MATE_BOUND == TT_MATE_BOUND

ENTER, NEXT, RETURN = 0, 1, 2
SC_SP, SC_MODE, SC_VALUE, SC_NODES, SC_SELDEPTH, SC_PREV_PV_LEN = range(6)

FOLLOW_PV = 1
"""Frame flag: every move from the root to this frame matches the previous iteration's PV."""

PV_SCORE = 2_000_000
CAPTURE_SCORE = 1_000_000

Int32Array = npt.NDArray[np.int32]


class SearchStack(NamedTuple):
    moves: Int32Array
    """(MAX_SEARCH_PLY, MAX_MOVES): move list of each frame."""
    order: Int32Array
    """(MAX_SEARCH_PLY, MAX_MOVES): ordering score of each move (higher is searched first)."""
    n_moves: Int32Array
    next_index: Int32Array
    """Next move to search in each frame."""
    alpha: Int32Array
    alpha0: Int32Array
    """Alpha when the frame was entered (``alpha`` rises as moves improve on it)."""
    beta: Int32Array
    best: Int32Array
    depth: Int32Array
    best_move: Int32Array
    flags: Int32Array
    pv: Int32Array
    """(MAX_SEARCH_PLY, MAX_SEARCH_PLY) triangular PV table; row ``p`` starts at column ``p``."""
    pv_len: Int32Array
    prev_pv: Int32Array
    """PV of the last completed iteration, followed first to order moves."""
    sc: npt.NDArray[np.int64]
    """Scalars: stack pointer, loop mode, pending return value, nodes, seldepth, prev PV length."""


def new_stack() -> SearchStack:
    n = MAX_SEARCH_PLY

    def row() -> Int32Array:
        return np.zeros(n, np.int32)

    return SearchStack(
        np.zeros((n, MAX_MOVES), np.int32),
        np.zeros((n, MAX_MOVES), np.int32),
        row(),
        row(),
        row(),
        row(),
        row(),
        row(),
        row(),
        row(),
        row(),
        np.zeros((n, n), np.int32),
        row(),
        row(),
        np.zeros(8, np.int64),
    )


@njit
def start_iteration(ss: SearchStack, depth: int, alpha: int, beta: int) -> None:
    """Set up the root frame for a ``depth``-ply search (the root position is ``pos`` as given)."""
    sc = ss.sc
    sc[SC_SP] = 0
    sc[SC_MODE] = ENTER
    sc[SC_VALUE] = 0
    sc[SC_SELDEPTH] = 0
    ss.alpha[0] = alpha
    ss.alpha0[0] = alpha
    ss.beta[0] = beta
    ss.depth[0] = depth
    ss.flags[0] = FOLLOW_PV
    ss.best_move[0] = 0
    ss.pv_len[0] = 0


@njit
def is_repetition(pos: Position) -> bool:
    """Whether the current position occurred before since the last irreversible move."""
    st = pos.st
    ply = st[ST_PLY]
    key = pos.keys[ply]
    stop = max(ply - st[ST_HALFMOVE], 0)
    i = ply - 4
    while i >= stop:
        if pos.keys[i] == key:
            return True
        i -= 2
    return False


@njit
def _score_moves(pos: Position, ss: SearchStack, p: int, n: int, tt_move: int) -> None:
    moves = ss.moves[p]
    order = ss.order[p]
    pv_move = -1
    if ss.flags[p] & FOLLOW_PV and p < ss.sc[SC_PREV_PV_LEN]:
        pv_move = ss.prev_pv[p]
    for i in range(n):
        m = moves[i]
        if m == pv_move:
            order[i] = PV_SCORE + 1
            continue
        if m == tt_move:
            order[i] = PV_SCORE
            continue
        flag = (m >> 12) & 15
        s = 0
        if flag & CAPTURE:
            victim = PAWN if flag == EP_CAPTURE else pos.board[(m >> 6) & 63] % 6
            attacker = pos.board[m & 63] % 6
            s = CAPTURE_SCORE + (victim + 1) * 8 - attacker
        if flag & PROMO:
            s = max(s, CAPTURE_SCORE) + (QUEEN + 1) * 8 if (flag & 3) + 1 == QUEEN else -1
        order[i] = s


@njit
def _enter(
    pos: Position, t: Tables, ev: EvalTables, ss: SearchStack, tt: TransTable, p: int
) -> tuple[bool, int]:
    """Set up frame ``p``; returns ``(True, score)`` if the node is a leaf, else ``(False, 0)``."""
    st = pos.st
    if p > 0 and (st[ST_HALFMOVE] >= 100 or is_repetition(pos) or insufficient_material(pos)):
        return True, 0
    if p >= MAX_SEARCH_PLY - 1:
        return True, evaluate(pos, ev)
    ss.alpha0[p] = ss.alpha[p]
    depth = ss.depth[p]
    key = pos.keys[st[ST_PLY]]
    ti = tt_index(tt, key)
    tt_move = 0
    if tt.keys[ti] == key:
        tt_move = int(tt.moves[ti])
        if p > 0 and ss.beta[p] - ss.alpha[p] == 1 and tt.depths[ti] >= max(depth, 0):
            score = score_from_tt(int(tt.scores[ti]), p)
            bound = tt.bounds[ti]
            if bound == BOUND_EXACT:
                return True, score
            if bound == BOUND_LOWER and score >= ss.beta[p]:
                return True, score
            if bound == BOUND_UPPER and score <= ss.alpha[p]:
                return True, score
    moves = ss.moves[p]
    check = in_check(pos, t)
    if depth <= 0 and not check:
        stand = evaluate(pos, ev)
        if stand >= ss.beta[p]:
            return True, stand
        if stand > ss.alpha[p]:
            ss.alpha[p] = stand
        n = generate_moves(pos, t, moves, True)
        kept = 0
        for i in range(n):
            m = moves[i]
            flag = (m >> 12) & 15
            if not (flag & PROMO) or (flag & 3) + 1 == QUEEN:
                moves[kept] = m
                kept += 1
        if kept == 0:
            return True, stand
        n = kept
        ss.best[p] = stand
    else:
        n = generate_moves(pos, t, moves, False)
        if n == 0:
            return True, (p - MATE) if check else 0
        ss.best[p] = -INF
    _score_moves(pos, ss, p, n, tt_move)
    ss.n_moves[p] = n
    ss.next_index[p] = 0
    ss.best_move[p] = 0
    return False, 0


@njit
def _store(pos: Position, ss: SearchStack, tt: TransTable, p: int, score: int) -> None:
    if score <= ss.alpha0[p]:
        bound = BOUND_UPPER
    elif score >= ss.beta[p]:
        bound = BOUND_LOWER
    else:
        bound = BOUND_EXACT
    move = ss.best_move[p] if bound != BOUND_UPPER else 0
    key = pos.keys[pos.st[ST_PLY]]
    tt_store(tt, key, move, score, 0, max(int(ss.depth[p]), 0), bound, p)


@njit
def run(
    pos: Position, t: Tables, ev: EvalTables, ss: SearchStack, tt: TransTable, budget: int
) -> bool:
    """Search up to ``budget`` more nodes; returns True once the root search has finished.

    When it returns True, ``ss.sc[SC_VALUE]`` is the root score, ``ss.pv[0, : ss.pv_len[0]]`` the
    principal variation, and ``pos`` is back at the root. When it returns False, ``pos`` is
    somewhere inside the tree: call ``run`` again to continue, or :func:`unwind` to abandon.
    """
    sc = ss.sc
    sp = int(sc[SC_SP])
    mode = int(sc[SC_MODE])
    value = int(sc[SC_VALUE])
    spent = 0
    finished = False
    while True:
        if mode == ENTER:
            if spent >= budget:
                break
            spent += 1
            ss.pv_len[sp] = 0
            if sp > sc[SC_SELDEPTH]:
                sc[SC_SELDEPTH] = sp
            leaf, value = _enter(pos, t, ev, ss, tt, sp)
            mode = RETURN if leaf else NEXT
        elif mode == NEXT:
            p = sp
            i = ss.next_index[p]
            n = ss.n_moves[p]
            if i >= n:
                value = ss.best[p]
                _store(pos, ss, tt, p, value)
                mode = RETURN
                continue
            order = ss.order[p]
            moves = ss.moves[p]
            j = i
            for k in range(i + 1, n):
                if order[k] > order[j]:
                    j = k
            m = moves[j]
            if j != i:
                moves[j], moves[i] = moves[i], m
                order[j], order[i] = order[i], order[j]
            ss.next_index[p] = i + 1
            make_move(pos, t, m)
            c = p + 1
            ss.alpha[c] = -ss.beta[p]
            ss.beta[c] = -ss.alpha[p]
            ss.depth[c] = ss.depth[p] - 1
            follow = ss.flags[p] & FOLLOW_PV and p < sc[SC_PREV_PV_LEN] and m == ss.prev_pv[p]
            ss.flags[c] = FOLLOW_PV if follow else 0
            sp = c
            mode = ENTER
        else:
            if sp == 0:
                finished = True
                break
            unmake_move(pos, t)
            sp -= 1
            p = sp
            score = -value
            mode = NEXT
            if score > ss.best[p]:
                ss.best[p] = score
                m = ss.moves[p, ss.next_index[p] - 1]
                ss.best_move[p] = m
                if score > ss.alpha[p]:
                    ss.alpha[p] = score
                    ss.pv[p, p] = m
                    cl = ss.pv_len[p + 1]
                    for k in range(cl):
                        ss.pv[p, p + 1 + k] = ss.pv[p + 1, p + 1 + k]
                    ss.pv_len[p] = cl + 1
                    if score >= ss.beta[p]:
                        value = score
                        _store(pos, ss, tt, p, value)
                        mode = RETURN
    sc[SC_SP] = sp
    sc[SC_MODE] = mode
    sc[SC_VALUE] = value
    sc[SC_NODES] += spent
    return finished


@njit
def unwind(pos: Position, t: Tables, ss: SearchStack) -> None:
    """Abandon an unfinished search and take ``pos`` back to the root."""
    while ss.sc[SC_SP] > 0:
        unmake_move(pos, t)
        ss.sc[SC_SP] -= 1
