"""Fail-soft negamax alpha-beta with quiescence, written as a resumable loop over an explicit stack.

numba segfaults when it reloads a cached self-recursive function, so the search keeps one frame
per ply in a :class:`SearchStack` instead of recursing. The same property makes it resumable:
:func:`run` searches at most ``budget`` nodes and returns, leaving the position and the stack mid-
tree, and the next call carries on. The Python driver uses this to read the clock and react to
``stop`` between chunks without the jitted code needing a clock of its own.

A frame whose remaining depth is 0 or less is a quiescence node: stand pat, then captures and
queen promotions only (or every evasion when in check).

A side in check at a full-width frame is searched one ply deeper (check extension), as long as the
frame is within twice the iteration depth of the root.

Moves after the first are searched with a null window around alpha (principal variation search);
a move that lands strictly inside the frame's window is searched again with the full window. The
re-search is a stage of the parent frame (``SearchStack.zw``), so it needs no recursion either.

Null-move pruning is the same kind of parent-frame stage: a non-PV side that is not in check, has
a piece besides king and pawns, and whose static eval is already at least beta may pass the move
and search the reply at a reduced depth. A fail high prunes the node. Consecutive null moves are
not allowed, and pawn endings are skipped so zugzwang is not mistaken for a cutoff.

Reverse futility pruning returns the static eval at a non-PV node whose eval is already at least
beta plus a margin that grows with remaining depth. Futility pruning skips later quiet moves at
those nodes when even a generous margin on the static eval cannot raise alpha; checks are still
searched.

Late-move reductions search later quiet non-checks at a shallower depth with a null window, then
re-search at full depth if that fails high. The re-search is another parent-frame stage. The root
is not reduced, so a quiet mate at the horizon is still searched at full iteration depth.
Late-move pruning drops later quiet non-checks at shallow non-PV nodes. Main-search SEE pruning
skips later losing captures and hanging quiets; quiescence already dropped losing captures.

Every completed frame is stored in the transposition table, and non-PV frames (null window) return
straight from a table entry that is deep enough and whose bound proves the result. The table move
is searched first.
"""

from __future__ import annotations

from math import log
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import BISHOP, KNIGHT, PAWN, QUEEN, ROOK, ZERO, Tables
from chessbot.core.jit import njit
from chessbot.core.movegen import MAX_MOVES, generate_moves, in_check
from chessbot.core.position import (
    CAPTURE,
    EP_CAPTURE,
    PROMO,
    ST_HALFMOVE,
    ST_PLY,
    ST_SIDE,
    U_MOVE,
    Position,
    make_move,
    make_null_move,
    unmake_move,
    unmake_null_move,
)
from chessbot.search.evaluate import EvalTables, evaluate, insufficient_material
from chessbot.search.see import SEE_BUFFER, see, see_value
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
FLAG_IN_CHECK = 2
FLAG_NMP = 4
"""Set on a frame that still has to search a null move before its real moves."""

CHILD_FULL, CHILD_PVS, CHILD_NMP, CHILD_LMR = 0, 1, 2, 3
"""What ``SearchStack.zw[p]`` says about the child currently open under frame ``p``."""

NMP_MIN_DEPTH = 3
"""Null-move pruning is not tried at remaining depths below this."""
RFP_MAX_DEPTH = 8
RFP_MARGIN = 80
"""Reverse futility: a non-PV node with eval - 80*depth already above beta is a cutoff."""
FP_MAX_DEPTH = 8
FP_MARGIN_BASE = 150
FP_MARGIN_PLY = 80
"""Futility: skip a quiet when eval + 150 + 80*depth cannot raise alpha."""
LMR_MIN_DEPTH = 3
LMR_DIVISOR = 2.25
"""LMR reduction is ``log(depth) * log(move_index+1) / 2.25``, plus one ply off the PV."""
LMP_MAX_DEPTH = 6
LMP_BASE = 3
"""Late-move pruning drops quiets from index ``3 + depth*depth`` at shallow non-PV nodes."""
SEE_PRUNE_MAX_DEPTH = 8
SEE_CAPTURE_MARGIN = 100
SEE_QUIET_MARGIN = 20
"""Main-search SEE: skip captures losing more than 100*depth and quiets hanging more than 20*d^2."""

PV_SCORE = 2_000_000
CAPTURE_SCORE = 1_000_000
BAD_CAPTURE_SCORE = -1_000_000
"""Captures that lose material (by SEE) are searched after every quiet move."""
UNDERPROMOTION_SCORE = -1_500_000
KILLER_SCORES = (900_000, 800_000)
COUNTER_SCORE = 700_000
HISTORY_MAX = 16384
"""History scores stay within +/- this, so quiet moves sort between killers and bad captures."""

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
    zw: Int32Array
    """Kind of child open under this frame: full, PVS, null-move, or LMR."""
    pv: Int32Array
    """(MAX_SEARCH_PLY, MAX_SEARCH_PLY) triangular PV table; row ``p`` starts at column ``p``."""
    pv_len: Int32Array
    prev_pv: Int32Array
    """PV of the last completed iteration, followed first to order moves."""
    sc: npt.NDArray[np.int64]
    """Scalars: stack pointer, loop mode, pending return value, nodes, seldepth, prev PV length."""
    see_buf: Int32Array
    killers: Int32Array
    """(MAX_SEARCH_PLY, 2): the last two quiet moves that caused a beta cutoff at each ply."""
    history: Int32Array
    """(2, 64, 64): side, from, to; rewarded on quiet cutoffs, penalised for quiets that failed."""
    counter: Int32Array
    """(12 * 64,): quiet refutation of the previous move, by its piece and destination."""
    static_eval: Int32Array
    """Side-to-move static evaluation of this frame, or 0 if it was never computed."""


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
        row(),
        np.zeros((n, n), np.int32),
        row(),
        row(),
        np.zeros(8, np.int64),
        np.zeros(SEE_BUFFER, np.int32),
        np.zeros((n, 2), np.int32),
        np.zeros((2, 64, 64), np.int32),
        np.zeros(12 * 64, np.int32),
        row(),
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
def _counter_slot(pos: Position) -> int:
    """Index into ``SearchStack.counter`` for the move that led here, or -1 after a null move or
    at the start of the game record."""
    ply = pos.st[ST_PLY]
    if ply == 0:
        return -1
    prev = pos.undo[ply - 1, U_MOVE]
    if prev == 0:
        return -1
    to = int(prev >> 6) & 63
    return int(pos.board[to]) * 64 + to


@njit
def _history_add(hist: Int32Array, m: int, bonus: int) -> None:
    frm, to = m & 63, (m >> 6) & 63
    h = hist[frm, to]
    hist[frm, to] = h + bonus - h * abs(bonus) // HISTORY_MAX


@njit
def _quiet_cutoff(pos: Position, ss: SearchStack, p: int, m: int) -> None:
    """Reward quiet move ``m`` for a beta cutoff at frame ``p`` and penalise the quiet moves
    searched before it."""
    if ss.killers[p, 0] != m:
        ss.killers[p, 1] = ss.killers[p, 0]
        ss.killers[p, 0] = m
    counter = _counter_slot(pos)
    if counter >= 0:
        ss.counter[counter] = m
    depth = ss.depth[p]
    bonus = min(300 * depth - 250, 1500)
    hist = ss.history[pos.st[ST_SIDE]]
    _history_add(hist, m, bonus)
    moves = ss.moves[p]
    for k in range(ss.next_index[p] - 1):
        q = moves[k]
        if not ((q >> 12) & (CAPTURE | PROMO)):
            _history_add(hist, q, -bonus)


@njit
def _score_moves(pos: Position, t: Tables, ss: SearchStack, p: int, n: int, tt_move: int) -> None:
    moves = ss.moves[p]
    order = ss.order[p]
    pv_move = -1
    if ss.flags[p] & FOLLOW_PV and p < ss.sc[SC_PREV_PV_LEN]:
        pv_move = ss.prev_pv[p]
    killer0, killer1 = ss.killers[p, 0], ss.killers[p, 1]
    counter = _counter_slot(pos)
    counter_move = ss.counter[counter] if counter >= 0 else -1
    hist = ss.history[pos.st[ST_SIDE]]
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
        if flag & PROMO and (flag & 3) + 1 != QUEEN:
            s = UNDERPROMOTION_SCORE
        elif flag & CAPTURE:
            victim = PAWN if flag == EP_CAPTURE else pos.board[(m >> 6) & 63] % 6
            attacker = pos.board[m & 63] % 6
            mvv_lva = (victim + 1) * 8 - attacker
            if flag & PROMO:
                mvv_lva += (QUEEN + 1) * 8
            losing = see_value(attacker) > see_value(victim) and see(pos, t, m, ss.see_buf) < 0
            s = (BAD_CAPTURE_SCORE if losing else CAPTURE_SCORE) + mvv_lva
        elif flag & PROMO:
            s = CAPTURE_SCORE + (QUEEN + 1) * 8
        elif m == killer0:
            s = KILLER_SCORES[0]
        elif m == killer1:
            s = KILLER_SCORES[1]
        elif m == counter_move:
            s = COUNTER_SCORE
        else:
            s = hist[m & 63, (m >> 6) & 63]
        order[i] = s


@njit
def _has_non_pawn(pos: Position) -> bool:
    """Whether the side to move has a piece other than king and pawns (NMP zugzwang guard)."""
    us = int(pos.st[ST_SIDE])
    bb = pos.bb
    pieces = bb[us * 6 + KNIGHT] | bb[us * 6 + BISHOP] | bb[us * 6 + ROOK] | bb[us * 6 + QUEEN]
    return bool(pieces != ZERO)


@njit
def _can_null_move(
    pos: Position, ss: SearchStack, p: int, check: bool, depth: int, stand: int
) -> bool:
    if p == 0 or check or depth < NMP_MIN_DEPTH:
        return False
    if ss.beta[p] - ss.alpha[p] > 1:
        return False
    if stand < ss.beta[p] or ss.beta[p] >= MATE_BOUND or ss.beta[p] <= -MATE_BOUND:
        return False
    if not _has_non_pawn(pos):
        return False
    ply = int(pos.st[ST_PLY])
    return not (ply > 0 and pos.undo[ply - 1, U_MOVE] == 0)


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
    if p > 0:
        md_alpha = -MATE + p
        md_beta = MATE - p - 1
        if ss.alpha[p] < md_alpha:
            ss.alpha[p] = md_alpha
            ss.alpha0[p] = md_alpha
        if ss.beta[p] > md_beta:
            ss.beta[p] = md_beta
        if ss.alpha[p] >= ss.beta[p]:
            return True, ss.alpha[p]
    check = in_check(pos, t)
    if check:
        ss.flags[p] |= FLAG_IN_CHECK
    depth = ss.depth[p]
    if check and depth >= 0 and 0 < p < 2 * ss.depth[0]:
        depth += 1
        ss.depth[p] = depth
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
    stand = 0
    if depth <= 0 and not check:
        stand = evaluate(pos, ev)
        ss.static_eval[p] = stand
        if stand >= ss.beta[p]:
            return True, stand
        if stand > ss.alpha[p]:
            ss.alpha[p] = stand
        n = generate_moves(pos, t, moves, True)
        kept = 0
        for i in range(n):
            m = moves[i]
            flag = (m >> 12) & 15
            if flag & PROMO and (flag & 3) + 1 != QUEEN:
                continue
            if (
                flag & CAPTURE
                and see_value(pos.board[m & 63] % 6) > see_value(pos.board[(m >> 6) & 63] % 6)
                and see(pos, t, m, ss.see_buf) < 0
            ):
                continue
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
        stand = 0 if check else evaluate(pos, ev)
        ss.static_eval[p] = stand
        ss.best[p] = -INF
        if (
            p > 0
            and not check
            and 0 < depth <= RFP_MAX_DEPTH
            and ss.beta[p] - ss.alpha[p] == 1
            and abs(ss.beta[p]) < MATE_BOUND
            and stand - RFP_MARGIN * depth >= ss.beta[p]
        ):
            return True, stand
        if _can_null_move(pos, ss, p, check, depth, stand):
            ss.flags[p] |= FLAG_NMP
    _score_moves(pos, t, ss, p, n, tt_move)
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
    tt_store(tt, key, move, score, int(ss.static_eval[p]), max(int(ss.depth[p]), 0), bound, p)


@njit
def _open_child(
    pos: Position,
    t: Tables,
    ss: SearchStack,
    p: int,
    m: int,
    null_window: bool,
    reduction: int,
) -> int:
    """Play ``m`` from frame ``p`` and set up the child frame's window; returns its index."""
    make_move(pos, t, m)
    c = p + 1
    if reduction > 0:
        ss.zw[p] = CHILD_LMR
    elif null_window:
        ss.zw[p] = CHILD_PVS
    else:
        ss.zw[p] = CHILD_FULL
    ss.alpha[c] = -ss.alpha[p] - 1 if null_window else -ss.beta[p]
    ss.beta[c] = -ss.alpha[p]
    ss.depth[c] = ss.depth[p] - 1 - reduction
    follow = ss.flags[p] & FOLLOW_PV and p < ss.sc[SC_PREV_PV_LEN] and m == ss.prev_pv[p]
    ss.flags[c] = FOLLOW_PV if follow else 0
    return c


@njit
def _open_null(pos: Position, t: Tables, ss: SearchStack, p: int, reduction: int) -> int:
    """Pass the move at frame ``p`` and search the reply at ``depth - reduction``."""
    make_null_move(pos, t)
    c = p + 1
    ss.zw[p] = CHILD_NMP
    ss.alpha[c] = -ss.beta[p]
    ss.beta[c] = -ss.beta[p] + 1
    ss.depth[c] = ss.depth[p] - reduction
    ss.flags[c] = 0
    return c


@njit
def _unmake_child(pos: Position, t: Tables, ss: SearchStack, p: int) -> None:
    if ss.zw[p] == CHILD_NMP:
        unmake_null_move(pos, t)
    else:
        unmake_move(pos, t)


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
            if leaf:
                mode = RETURN
            elif ss.flags[sp] & FLAG_NMP:
                reduction = 3 + int(ss.depth[sp]) // 6
                sp = _open_null(pos, t, ss, sp, reduction)
                mode = ENTER
            else:
                mode = NEXT
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
            flag = (m >> 12) & 15
            quiet = (flag & (CAPTURE | PROMO)) == 0
            depth_p = int(ss.depth[p])
            is_pv = ss.beta[p] - ss.alpha[p] > 1
            in_chk = ss.flags[p] & FLAG_IN_CHECK
            gives_check = False
            if i > 0 and depth_p > 0 and not in_chk:
                make_move(pos, t, m)
                gives_check = in_check(pos, t)
                unmake_move(pos, t)
            if (
                quiet
                and i > 0
                and 0 < depth_p <= FP_MAX_DEPTH
                and not is_pv
                and not in_chk
                and not gives_check
                and abs(ss.alpha[p]) < MATE_BOUND
                and ss.static_eval[p] + FP_MARGIN_BASE + FP_MARGIN_PLY * depth_p <= ss.alpha[p]
            ):
                continue
            if (
                quiet
                and i > 0
                and 0 < depth_p <= LMP_MAX_DEPTH
                and not is_pv
                and not in_chk
                and not gives_check
                and i >= LMP_BASE + depth_p * depth_p
            ):
                continue
            if (
                i > 0
                and 0 < depth_p <= SEE_PRUNE_MAX_DEPTH
                and not is_pv
                and not in_chk
                and not gives_check
            ):
                see_gain = see(pos, t, m, ss.see_buf)
                if flag & CAPTURE:
                    if see_gain < -SEE_CAPTURE_MARGIN * depth_p:
                        continue
                elif quiet and see_gain < -SEE_QUIET_MARGIN * depth_p * depth_p:
                    continue
            reduction = 0
            if (
                p > 0
                and quiet
                and i > 0
                and depth_p >= LMR_MIN_DEPTH
                and not in_chk
                and not gives_check
            ):
                r = int(log(float(depth_p)) * log(float(i + 1)) / LMR_DIVISOR)
                if not is_pv:
                    r += 1
                if r < 1:
                    r = 1
                if r > depth_p - 1:
                    r = depth_p - 1
                reduction = r
            null_window = (i > 0 and is_pv) or reduction > 0
            sp = _open_child(pos, t, ss, p, m, null_window, reduction)
            mode = ENTER
        else:
            if sp == 0:
                finished = True
                break
            p = sp - 1
            _unmake_child(pos, t, ss, p)
            sp = p
            score = -value
            if ss.zw[p] == CHILD_NMP:
                ss.zw[p] = CHILD_FULL
                ss.flags[p] &= ~FLAG_NMP
                if score >= ss.beta[p]:
                    if score >= MATE_BOUND:
                        score = ss.beta[p]
                    ss.best[p] = score
                    value = score
                    _store(pos, ss, tt, p, value)
                    mode = RETURN
                else:
                    mode = NEXT
                continue
            if ss.zw[p] == CHILD_LMR and score > ss.alpha[p]:
                m = ss.moves[p, ss.next_index[p] - 1]
                i = int(ss.next_index[p] - 1)
                is_pv = ss.beta[p] - ss.alpha[p] > 1
                sp = _open_child(pos, t, ss, p, m, i > 0 and is_pv, 0)
                mode = ENTER
                continue
            mode = NEXT
            if ss.zw[p] == CHILD_PVS and ss.alpha[p] < score < ss.beta[p]:
                sp = _open_child(pos, t, ss, p, ss.moves[p, ss.next_index[p] - 1], False, 0)
                mode = ENTER
                continue
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
                        if ss.depth[p] > 0 and not ((m >> 12) & (CAPTURE | PROMO)):
                            _quiet_cutoff(pos, ss, p, m)
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
        p = int(ss.sc[SC_SP]) - 1
        _unmake_child(pos, t, ss, p)
        ss.sc[SC_SP] = p
