"""Legal move generation with check and pin masks (no make/unmake legality test per move).

``generate_moves`` writes moves into a caller-provided ``int32`` buffer of at least
``MAX_MOVES`` entries and returns how many it wrote. With ``tactical_only`` it emits only
captures (including en passant) and promotions, for quiescence search.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import (
    BISHOP,
    FULL,
    KING,
    KING_ATT,
    KNIGHT,
    KNIGHT_ATT,
    NO_PIECE,
    NO_SQ,
    ONE,
    PAWN,
    QUEEN,
    RANK_1,
    RANK_2,
    RANK_7,
    RANK_8,
    ROOK,
    U64,
    WHITE,
    WHITE_OCC,
    ZERO,
    Tables,
    bishop_attacks,
    bit,
    lsb,
    popcount,
    rook_attacks,
)
from chessbot.core.jit import njit
from chessbot.core.position import (
    CAPTURE,
    DOUBLE_PUSH,
    EP_CAPTURE,
    KING_CASTLE,
    PROMO,
    QUEEN_CASTLE,
    QUIET,
    ST_CASTLE,
    ST_EP,
    ST_SIDE,
    Position,
)

MAX_MOVES = 256

MoveBuffer = npt.NDArray[np.int32]

WK_PATH = U64(0x60)
WQ_PATH = U64(0x0E)
BK_PATH = U64(0x60 << 56)
BQ_PATH = U64(0x0E << 56)


@njit
def attackers_to(
    t: Tables, bb: npt.NDArray[np.uint64], sq: int, occ: np.uint64, by: int
) -> np.uint64:
    """Pieces of colour ``by`` attacking ``sq`` given occupancy ``occ``."""
    o = by * 6
    diag = bb[o + BISHOP] | bb[o + QUEEN]
    orth = bb[o + ROOK] | bb[o + QUEEN]
    return (  # type: ignore[no-any-return]
        (t.leaper[by ^ 1, sq] & bb[o + PAWN])
        | (t.leaper[KNIGHT_ATT, sq] & bb[o + KNIGHT])
        | (t.leaper[KING_ATT, sq] & bb[o + KING])
        | (bishop_attacks(t, sq, occ) & diag)
        | (rook_attacks(t, sq, occ) & orth)
    )


@njit
def in_check(pos: Position, t: Tables) -> bool:
    bb = pos.bb
    us = pos.st[ST_SIDE]
    occ = bb[WHITE_OCC] | bb[WHITE_OCC + 1]
    return bool(attackers_to(t, bb, lsb(bb[us * 6 + KING]), occ, us ^ 1) != ZERO)


@njit
def _add_targets(moves: MoveBuffer, n: int, frm: int, targets: np.uint64, opp: np.uint64) -> int:
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        flag = CAPTURE if opp & bit(to) else QUIET
        moves[n] = frm | (to << 6) | (flag << 12)
        n += 1
    return n


@njit
def _add_promotions(moves: MoveBuffer, n: int, frm: int, to: int, capture: int) -> int:
    for p in range(3, -1, -1):
        moves[n] = frm | (to << 6) | ((PROMO | capture | p) << 12)
        n += 1
    return n


@njit
def generate_moves(pos: Position, t: Tables, moves: MoveBuffer, tactical_only: bool) -> int:
    bb, st = pos.bb, pos.st
    us = st[ST_SIDE]
    them = us ^ 1
    o = us * 6
    e = them * 6
    own = bb[WHITE_OCC + us]
    opp = bb[WHITE_OCC + them]
    occ = own | opp
    ksq = lsb(bb[o + KING])
    checkers = attackers_to(t, bb, ksq, occ, them)
    n = 0

    target_filter = opp if tactical_only else ~own
    kocc = occ ^ bit(ksq)
    targets = t.leaper[KING_ATT, ksq] & target_filter
    while targets:
        to = lsb(targets)
        targets &= targets - ONE
        if attackers_to(t, bb, to, kocc, them) == ZERO:
            flag = CAPTURE if opp & bit(to) else QUIET
            moves[n] = ksq | (to << 6) | (flag << 12)
            n += 1

    nchecks = popcount(checkers)
    if nchecks > 1:
        return n

    if nchecks == 1:
        check_mask = t.between[ksq, lsb(checkers)] | checkers
    else:
        check_mask = FULL
        if not tactical_only:
            castle = st[ST_CASTLE]
            if us == WHITE:
                if (
                    castle & 1
                    and not (occ & WK_PATH)
                    and attackers_to(t, bb, 5, occ, them) == ZERO
                    and attackers_to(t, bb, 6, occ, them) == ZERO
                ):
                    moves[n] = 4 | (6 << 6) | (KING_CASTLE << 12)
                    n += 1
                if (
                    castle & 2
                    and not (occ & WQ_PATH)
                    and attackers_to(t, bb, 3, occ, them) == ZERO
                    and attackers_to(t, bb, 2, occ, them) == ZERO
                ):
                    moves[n] = 4 | (2 << 6) | (QUEEN_CASTLE << 12)
                    n += 1
            else:
                if (
                    castle & 4
                    and not (occ & BK_PATH)
                    and attackers_to(t, bb, 61, occ, them) == ZERO
                    and attackers_to(t, bb, 62, occ, them) == ZERO
                ):
                    moves[n] = 60 | (62 << 6) | (KING_CASTLE << 12)
                    n += 1
                if (
                    castle & 8
                    and not (occ & BQ_PATH)
                    and attackers_to(t, bb, 59, occ, them) == ZERO
                    and attackers_to(t, bb, 58, occ, them) == ZERO
                ):
                    moves[n] = 60 | (58 << 6) | (QUEEN_CASTLE << 12)
                    n += 1

    pinned = ZERO
    snipers = (bishop_attacks(t, ksq, opp) & (bb[e + BISHOP] | bb[e + QUEEN])) | (
        rook_attacks(t, ksq, opp) & (bb[e + ROOK] | bb[e + QUEEN])
    )
    while snipers:
        s = lsb(snipers)
        snipers &= snipers - ONE
        blockers = t.between[ksq, s] & occ
        if blockers and not (blockers & (blockers - ONE)) and blockers & own:
            pinned |= blockers

    mask = check_mask & target_filter

    pcs = bb[o + KNIGHT] & ~pinned
    while pcs:
        frm = lsb(pcs)
        pcs &= pcs - ONE
        n = _add_targets(moves, n, frm, t.leaper[KNIGHT_ATT, frm] & mask, opp)

    pcs = bb[o + BISHOP] | bb[o + QUEEN]
    while pcs:
        frm = lsb(pcs)
        pcs &= pcs - ONE
        targets = bishop_attacks(t, frm, occ) & mask
        if pinned & bit(frm):
            targets &= t.line[ksq, frm]
        n = _add_targets(moves, n, frm, targets, opp)

    pcs = bb[o + ROOK] | bb[o + QUEEN]
    while pcs:
        frm = lsb(pcs)
        pcs &= pcs - ONE
        targets = rook_attacks(t, frm, occ) & mask
        if pinned & bit(frm):
            targets &= t.line[ksq, frm]
        n = _add_targets(moves, n, frm, targets, opp)

    if us == WHITE:
        push, start_rank, last_rank = 8, RANK_2, RANK_8
    else:
        push, start_rank, last_rank = -8, RANK_7, RANK_1
    ep = st[ST_EP]
    pcs = bb[o + PAWN]
    while pcs:
        frm = lsb(pcs)
        pcs &= pcs - ONE
        frm_bb = bit(frm)
        allowed = check_mask
        if pinned & frm_bb:
            allowed &= t.line[ksq, frm]
        to = frm + push
        if pos.board[to] == NO_PIECE:
            to_bb = bit(to)
            if to_bb & allowed:
                if to_bb & last_rank:
                    n = _add_promotions(moves, n, frm, to, 0)
                elif not tactical_only:
                    moves[n] = frm | (to << 6)
                    n += 1
            if not tactical_only and frm_bb & start_rank:
                to2 = to + push
                if pos.board[to2] == NO_PIECE and bit(to2) & allowed:
                    moves[n] = frm | (to2 << 6) | (DOUBLE_PUSH << 12)
                    n += 1
        pawn_att = t.leaper[us, frm]
        caps = pawn_att & opp & allowed
        while caps:
            to = lsb(caps)
            caps &= caps - ONE
            if bit(to) & last_rank:
                n = _add_promotions(moves, n, frm, to, CAPTURE)
            else:
                moves[n] = frm | (to << 6) | (CAPTURE << 12)
                n += 1
        if ep != NO_SQ and pawn_att & bit(ep):
            capsq = ep - push
            occ2 = (occ ^ frm_bb ^ bit(capsq)) | bit(ep)
            attacked = (
                (t.leaper[us, ksq] & bb[e + PAWN] & ~bit(capsq))
                | (t.leaper[KNIGHT_ATT, ksq] & bb[e + KNIGHT])
                | (bishop_attacks(t, ksq, occ2) & (bb[e + BISHOP] | bb[e + QUEEN]))
                | (rook_attacks(t, ksq, occ2) & (bb[e + ROOK] | bb[e + QUEEN]))
            )
            if attacked == ZERO:
                moves[n] = frm | (ep << 6) | (EP_CAPTURE << 12)
                n += 1
    return n
