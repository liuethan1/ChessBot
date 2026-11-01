"""Tapered evaluation: PeSTO piece-square tables plus Texel-tuned hand-crafted terms.

The PeSTO tables are Ronald Friederich's, as published on the CPW, written from White's point of
view with a8 first. They are folded into one middlegame and one endgame table indexed by
``[piece code, square]`` (our squares run a1 = 0 ... h8 = 63), with the material values included.

On top of that, :func:`evaluate` adds bishop pair, pawn-structure, mobility, rook-file, outpost,
and king-safety terms. Those weights start from literature values and are then Texel-tuned
(``chessbot-texel``) on quiet positions labelled with game results. The search still sees a
single tapered centipawn score from the side to move.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import (
    BISHOP,
    BLACK,
    FULL,
    KING,
    KING_ATT,
    KNIGHT,
    KNIGHT_ATT,
    ONE,
    PAWN,
    QUEEN,
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
from chessbot.core.position import ST_SIDE, Position

FILE_A = U64(0x0101010101010101)
FILE_H = U64(0x8080808080808080)
NOT_A = FULL ^ FILE_A
NOT_H = FULL ^ FILE_H

MG_VALUE = (82, 337, 365, 477, 1025, 0)
EG_VALUE = (94, 281, 297, 512, 936, 0)
PHASE_INC = (0, 1, 1, 2, 4, 0)
MAX_PHASE = 24

# fmt: off
MG_PST = (
    (   0,   0,   0,   0,   0,   0,   0,   0,
       98, 134,  61,  95,  68, 126,  34, -11,
       -6,   7,  26,  31,  65,  56,  25, -20,
      -14,  13,   6,  21,  23,  12,  17, -23,
      -27,  -2,  -5,  12,  17,   6,  10, -25,
      -26,  -4,  -4, -10,   3,   3,  33, -12,
      -35,  -1, -20, -23, -15,  24,  38, -22,
        0,   0,   0,   0,   0,   0,   0,   0),
    (-167, -89, -34, -49,  61, -97, -15,-107,
      -73, -41,  72,  36,  23,  62,   7, -17,
      -47,  60,  37,  65,  84, 129,  73,  44,
       -9,  17,  19,  53,  37,  69,  18,  22,
      -13,   4,  16,  13,  28,  19,  21,  -8,
      -23,  -9,  12,  10,  19,  17,  25, -16,
      -29, -53, -12,  -3,  -1,  18, -14, -19,
     -105, -21, -58, -33, -17, -28, -19, -23),
    ( -29,   4, -82, -37, -25, -42,   7,  -8,
      -26,  16, -18, -13,  30,  59,  18, -47,
      -16,  37,  43,  40,  35,  50,  37,  -2,
       -4,   5,  19,  50,  37,  37,   7,  -2,
       -6,  13,  13,  26,  34,  12,  10,   4,
        0,  15,  15,  15,  14,  27,  18,  10,
        4,  15,  16,   0,   7,  21,  33,   1,
      -33,  -3, -14, -21, -13, -12, -39, -21),
    (  32,  42,  32,  51,  63,   9,  31,  43,
       27,  32,  58,  62,  80,  67,  26,  44,
       -5,  19,  26,  36,  17,  45,  61,  16,
      -24, -11,   7,  26,  24,  35,  -8, -20,
      -36, -26, -12,  -1,   9,  -7,   6, -23,
      -45, -25, -16, -17,   3,   0,  -5, -33,
      -44, -16, -20,  -9,  -1,  11,  -6, -71,
      -19, -13,   1,  17,  16,   7, -37, -26),
    ( -28,   0,  29,  12,  59,  44,  43,  45,
      -24, -39,  -5,   1, -16,  57,  28,  54,
      -13, -17,   7,   8,  29,  56,  47,  57,
      -27, -27, -16, -16,  -1,  17,  -2,   1,
       -9, -26,  -9, -10,  -2,  -4,   3,  -3,
      -14,   2, -11,  -2,  -5,   2,  14,   5,
      -35,  -8,  11,   2,   8,  15,  -3,   1,
       -1, -18,  -9,  10, -15, -25, -31, -50),
    ( -65,  23,  16, -15, -56, -34,   2,  13,
       29,  -1, -20,  -7,  -8,  -4, -38, -29,
       -9,  24,   2, -16, -20,   6,  22, -22,
      -17, -20, -12, -27, -30, -25, -14, -36,
      -49,  -1, -27, -39, -46, -44, -33, -51,
      -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
      -15,  36,  12, -54,   8, -28,  24,  14),
)

EG_PST = (
    (   0,   0,   0,   0,   0,   0,   0,   0,
      178, 173, 158, 134, 147, 132, 165, 187,
       94, 100,  85,  67,  56,  53,  82,  84,
       32,  24,  13,   5,  -2,   4,  17,  17,
       13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
       13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0),
    ( -58, -38, -13, -28, -31, -27, -63, -99,
      -25,  -8, -25,  -2,  -9, -25, -24, -52,
      -24, -20,  10,   9,  -1,  -9, -19, -41,
      -17,   3,  22,  22,  22,  11,   8, -18,
      -18,  -6,  16,  25,  16,  17,   4, -18,
      -23,  -3,  -1,  15,  10,  -3, -20, -22,
      -42, -20, -10,  -5,  -2, -20, -23, -44,
      -29, -51, -23, -15, -22, -18, -50, -64),
    ( -14, -21, -11,  -8,  -7,  -9, -17, -24,
       -8,  -4,   7, -12,  -3, -13,  -4, -14,
        2,  -8,   0,  -1,  -2,   6,   0,   4,
       -3,   9,  12,   9,  14,  10,   3,   2,
       -6,   3,  13,  19,   7,  10,  -3,  -9,
      -12,  -3,   8,  10,  13,   3,  -7, -15,
      -14, -18,  -7,  -1,   4,  -9, -15, -27,
      -23,  -9, -23,  -5,  -9, -16,  -5, -17),
    (  13,  10,  18,  15,  12,  12,   8,   5,
       11,  13,  13,  11,  -3,   3,   8,   3,
        7,   7,   7,   5,   4,  -3,  -5,  -3,
        4,   3,  13,   1,   2,   1,  -1,   2,
        3,   5,   8,   4,  -5,  -6,  -8, -11,
       -4,   0,  -5,  -1,  -7, -12,  -8, -16,
       -6,  -6,   0,   2,  -9,  -9, -11,  -3,
       -9,   2,   3,  -1,  -5, -13,   4, -20),
    (  -9,  22,  22,  27,  27,  19,  10,  20,
      -17,  20,  32,  41,  58,  25,  30,   0,
      -20,   6,   9,  49,  47,  35,  19,   9,
        3,  22,  24,  45,  57,  40,  57,  36,
      -18,  28,  19,  47,  31,  34,  39,  23,
      -16, -27,  15,   6,   9,  17,  10,   5,
      -22, -23, -30, -16, -16, -23, -36, -32,
      -33, -28, -22, -43,  -5, -32, -20, -41),
    ( -74, -35, -18, -18, -11,  15,   4, -17,
      -12,  17,  14,  17,  17,  38,  23,  11,
       10,  17,  23,  15,  20,  45,  44,  13,
       -8,  22,  24,  27,  26,  33,  26,   3,
      -18,  -4,  21,  24,  27,  23,   9, -11,
      -19,  -3,  11,  21,  23,  16,   7,  -9,
      -27, -11,   4,  13,  14,   4,  -5, -17,
      -53, -34, -21, -11, -28, -14, -24, -43),
)
# fmt: on

# Extra-term indices. Passed-pawn ranks 1-7 occupy I_PASSED .. I_PASSED + 6.
I_BISHOP_PAIR = 0
I_DOUBLED = 1
I_ISOLATED = 2
I_SUPPORTED = 3
I_PHALANX = 4
I_PASSED = 5
N_PASSED = 7
I_ROOK_OPEN = 12
I_ROOK_SEMI = 13
I_ROOK_SEVENTH = 14
I_KNIGHT_MOB = 15
I_BISHOP_MOB = 16
I_ROOK_MOB = 17
I_QUEEN_MOB = 18
I_KING_SHELTER = 19
I_KING_STORM = 20
I_KING_ATTACK = 21
I_KNIGHT_OUTPOST = 22
I_BISHOP_OUTPOST = 23
I_PAWN_ISLANDS = 24
I_TEMPO = 25
N_TERMS = 26

TERM_NAMES = (
    "bishop_pair",
    "doubled",
    "isolated",
    "supported",
    "phalanx",
    "passed_1",
    "passed_2",
    "passed_3",
    "passed_4",
    "passed_5",
    "passed_6",
    "passed_7",
    "rook_open",
    "rook_semi",
    "rook_seventh",
    "knight_mob",
    "bishop_mob",
    "rook_mob",
    "queen_mob",
    "king_shelter",
    "king_storm",
    "king_attack",
    "knight_outpost",
    "bishop_outpost",
    "pawn_islands",
    "tempo",
)

# Texel-tuned extra-term weights (quiet WDL from Phase 4 games). Passed pawns, rook on
# the 7th, king storm, and king attack stay at literature values: the quiet set never
# sees a 7th-rank passer, and PeSTO's pawn/king tables already overlap those terms.
W_MG = (
    20,
    -23,
    -11,
    16,
    8,
    0,
    8,
    16,
    32,
    56,
    88,
    130,
    16,
    4,
    18,
    1,
    3,
    1,
    1,
    12,
    -6,
    1,
    8,
    6,
    0,
    8,
)
W_EG = (
    40,
    -23,
    -5,
    0,
    0,
    4,
    12,
    24,
    48,
    80,
    120,
    180,
    4,
    20,
    28,
    3,
    1,
    0,
    3,
    0,
    -2,
    0,
    15,
    16,
    0,
    6,
)
ATTACK_W = (0, 2, 2, 3, 5, 0)

assert len(TERM_NAMES) == N_TERMS == len(W_MG) == len(W_EG)


class EvalTables(NamedTuple):
    mg: npt.NDArray[np.int32]
    """(12, 64): middlegame material plus piece-square value, from the piece owner's view."""
    eg: npt.NDArray[np.int32]
    phase: npt.NDArray[np.int32]
    """(12,): game-phase weight of each piece code."""
    w_mg: npt.NDArray[np.int32]
    """(N_TERMS,): extra-term middlegame weights in centipawns."""
    w_eg: npt.NDArray[np.int32]
    attack_w: npt.NDArray[np.int32]
    """(6,): king-attack units for P,N,B,R,Q,K."""
    file_bb: npt.NDArray[np.uint64]
    isolated: npt.NDArray[np.uint64]
    """(64,): adjacent files of each square, for isolated-pawn detection."""
    passed: npt.NDArray[np.uint64]
    """(2, 64): squares in front on this and neighbouring files (white, black)."""
    forward: npt.NDArray[np.uint64]
    """(2, 64): squares ahead on the same file."""
    shelter: npt.NDArray[np.uint64]
    """(2, 64): two ranks in front on this and neighbouring files."""
    scratch: npt.NDArray[np.int32]
    """(N_TERMS,): scratch features for the single-threaded evaluator."""


def _build_masks() -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
]:
    file_bb = np.zeros(8, np.uint64)
    file_a = np.uint64(0x0101010101010101)
    for f in range(8):
        file_bb[f] = file_a << np.uint64(f)
    isolated = np.zeros(64, np.uint64)
    passed = np.zeros((2, 64), np.uint64)
    forward = np.zeros((2, 64), np.uint64)
    shelter = np.zeros((2, 64), np.uint64)
    one = np.uint64(1)
    for sq in range(64):
        f, r = sq % 8, sq // 8
        iso = np.uint64(0)
        if f > 0:
            iso |= file_bb[f - 1]
        if f < 7:
            iso |= file_bb[f + 1]
        isolated[sq] = iso
        pw = pb = fw = fb = sw = sb = np.uint64(0)
        for rr in range(r + 1, 8):
            fw |= one << np.uint64(rr * 8 + f)
            for ff in range(max(0, f - 1), min(8, f + 2)):
                pw |= one << np.uint64(rr * 8 + ff)
        for rr in range(r):
            fb |= one << np.uint64(rr * 8 + f)
            for ff in range(max(0, f - 1), min(8, f + 2)):
                pb |= one << np.uint64(rr * 8 + ff)
        for df in (-1, 0, 1):
            ff = f + df
            if not 0 <= ff < 8:
                continue
            for dr in (1, 2):
                rr = r + dr
                if rr < 8:
                    sw |= one << np.uint64(rr * 8 + ff)
                rr = r - dr
                if rr >= 0:
                    sb |= one << np.uint64(rr * 8 + ff)
        passed[WHITE, sq] = pw
        passed[BLACK, sq] = pb
        forward[WHITE, sq] = fw
        forward[BLACK, sq] = fb
        shelter[WHITE, sq] = sw
        shelter[BLACK, sq] = sb
    return file_bb, isolated, passed, forward, shelter


def build_eval_tables(
    w_mg: tuple[int, ...] | npt.NDArray[np.int32] | None = None,
    w_eg: tuple[int, ...] | npt.NDArray[np.int32] | None = None,
) -> EvalTables:
    mg = np.zeros((12, 64), np.int32)
    eg = np.zeros((12, 64), np.int32)
    phase = np.zeros(12, np.int32)
    for p in range(6):
        for sq in range(64):
            # CPW tables start at a8, so White's square ``sq`` is row-flipped and Black's is not.
            mg[p, sq] = MG_VALUE[p] + MG_PST[p][sq ^ 56]
            eg[p, sq] = EG_VALUE[p] + EG_PST[p][sq ^ 56]
            mg[6 + p, sq] = MG_VALUE[p] + MG_PST[p][sq]
            eg[6 + p, sq] = EG_VALUE[p] + EG_PST[p][sq]
        phase[p] = phase[6 + p] = PHASE_INC[p]
    file_bb, isolated, passed, forward, shelter = _build_masks()
    return EvalTables(
        mg,
        eg,
        phase,
        np.array(W_MG if w_mg is None else w_mg, np.int32),
        np.array(W_EG if w_eg is None else w_eg, np.int32),
        np.array(ATTACK_W, np.int32),
        file_bb,
        isolated,
        passed,
        forward,
        shelter,
        np.zeros(N_TERMS, np.int32),
    )


_EVAL_TABLES: EvalTables | None = None


def eval_tables() -> EvalTables:
    global _EVAL_TABLES
    if _EVAL_TABLES is None:
        _EVAL_TABLES = build_eval_tables()
    return _EVAL_TABLES


@njit
def _pawn_attacks(pawns: np.uint64, color: int) -> np.uint64:
    if color == WHITE:
        return ((pawns & NOT_A) << U64(7)) | ((pawns & NOT_H) << U64(9))
    return ((pawns & NOT_H) >> U64(7)) | ((pawns & NOT_A) >> U64(9))


@njit
def _islands(pawns: np.uint64, file_bb: npt.NDArray[np.uint64]) -> int:
    n = 0
    prev = 0
    for f in range(8):
        occ = 1 if (pawns & file_bb[f]) != ZERO else 0
        if occ and not prev:
            n += 1
        prev = occ
    return n


@njit
def _piece_attacks(t: Tables, pt: int, sq: int, occ: np.uint64) -> np.uint64:
    if pt == KNIGHT:
        return t.leaper[KNIGHT_ATT, sq]  # type: ignore[no-any-return]
    if pt == KING:
        return t.leaper[KING_ATT, sq]  # type: ignore[no-any-return]
    if pt == BISHOP:
        return bishop_attacks(t, sq, occ)
    if pt == ROOK:
        return rook_attacks(t, sq, occ)
    if pt == QUEEN:
        return bishop_attacks(t, sq, occ) | rook_attacks(t, sq, occ)
    return ZERO


@njit
def eval_components(
    pos: Position, t: Tables, ev: EvalTables, feat: npt.NDArray[np.int32]
) -> tuple[int, int, int]:
    """Fill ``feat`` with white-minus-black extra terms; return PeSTO (mg, eg, phase)."""
    for i in range(N_TERMS):
        feat[i] = 0
    bb = pos.bb
    occ = bb[WHITE_OCC] | bb[WHITE_OCC + 1]
    wp, bp = bb[PAWN], bb[6 + PAWN]
    own_w, own_b = bb[WHITE_OCC], bb[WHITE_OCC + 1]
    w_att = _pawn_attacks(wp, WHITE)
    b_att = _pawn_attacks(bp, BLACK)
    wking = lsb(bb[KING])
    bking = lsb(bb[6 + KING])
    zone_w = t.leaper[KING_ATT, wking] | bit(wking)
    zone_b = t.leaper[KING_ATT, bking] | bit(bking)
    units_w = 0
    units_b = 0

    pawns_w = wp
    while pawns_w:
        sq = lsb(pawns_w)
        pawns_w &= pawns_w - ONE
        r = sq >> 3
        if (ev.forward[WHITE, sq] & wp) != ZERO:
            feat[I_DOUBLED] += 1
        if (ev.isolated[sq] & wp) == ZERO:
            feat[I_ISOLATED] += 1
        if (ev.passed[WHITE, sq] & bp) == ZERO and 1 <= r <= 7:
            feat[I_PASSED + r - 1] += 1
        if (w_att & bit(sq)) != ZERO:
            feat[I_SUPPORTED] += 1
        if (sq & 7) < 7 and (wp & bit(sq + 1)) != ZERO:
            feat[I_PHALANX] += 1
        if (t.leaper[WHITE, sq] & zone_b) != ZERO:
            units_b += 1

    pawns_b = bp
    while pawns_b:
        sq = lsb(pawns_b)
        pawns_b &= pawns_b - ONE
        r = sq >> 3
        rel = 7 - r
        if (ev.forward[BLACK, sq] & bp) != ZERO:
            feat[I_DOUBLED] -= 1
        if (ev.isolated[sq] & bp) == ZERO:
            feat[I_ISOLATED] -= 1
        if (ev.passed[BLACK, sq] & wp) == ZERO and 1 <= rel <= 7:
            feat[I_PASSED + rel - 1] -= 1
        if (b_att & bit(sq)) != ZERO:
            feat[I_SUPPORTED] -= 1
        if (sq & 7) < 7 and (bp & bit(sq + 1)) != ZERO:
            feat[I_PHALANX] -= 1
        if (t.leaper[BLACK, sq] & zone_w) != ZERO:
            units_w += 1

    mg = 0
    eg = 0
    phase = 0
    for pc in range(12):
        color = 0 if pc < 6 else 1
        pt = pc - color * 6
        sign = 1 if color == 0 else -1
        own = own_w if color == 0 else own_b
        enemy_pawn_att = b_att if color == 0 else w_att
        own_pawn_att = w_att if color == 0 else b_att
        zone_them = zone_b if color == 0 else zone_w
        b = bb[pc]
        while b:
            sq = lsb(b)
            b &= b - ONE
            mg += sign * ev.mg[pc, sq]
            eg += sign * ev.eg[pc, sq]
            phase += ev.phase[pc]
            if pt in (PAWN, KING):
                continue
            att = _piece_attacks(t, pt, sq, occ)
            if (att & zone_them) != ZERO:
                if color == 0:
                    units_b += ev.attack_w[pt]
                else:
                    units_w += ev.attack_w[pt]
            if pt == KNIGHT:
                mob = popcount(att & ~own & ~enemy_pawn_att)
                feat[I_KNIGHT_MOB] += sign * mob
                rel = (sq >> 3) if color == 0 else 7 - (sq >> 3)
                if (
                    3 <= rel <= 5
                    and (own_pawn_att & bit(sq)) != ZERO
                    and (enemy_pawn_att & bit(sq)) == ZERO
                ):
                    feat[I_KNIGHT_OUTPOST] += sign
            elif pt == BISHOP:
                mob = popcount(att & ~own & ~enemy_pawn_att)
                feat[I_BISHOP_MOB] += sign * mob
                rel = (sq >> 3) if color == 0 else 7 - (sq >> 3)
                if (
                    3 <= rel <= 5
                    and (own_pawn_att & bit(sq)) != ZERO
                    and (enemy_pawn_att & bit(sq)) == ZERO
                ):
                    feat[I_BISHOP_OUTPOST] += sign
            elif pt == ROOK:
                feat[I_ROOK_MOB] += sign * popcount(att & ~own)
                f = sq & 7
                own_file = (wp if color == 0 else bp) & ev.file_bb[f]
                opp_file = (bp if color == 0 else wp) & ev.file_bb[f]
                if own_file == ZERO:
                    if opp_file == ZERO:
                        feat[I_ROOK_OPEN] += sign
                    else:
                        feat[I_ROOK_SEMI] += sign
                rel = (sq >> 3) if color == 0 else 7 - (sq >> 3)
                enemy_king_rank = (bking >> 3) if color == 0 else (wking >> 3)
                back = 7 if color == 0 else 0
                if rel == 6 and enemy_king_rank == back:
                    feat[I_ROOK_SEVENTH] += sign
            elif pt == QUEEN:
                feat[I_QUEEN_MOB] += sign * popcount(att & ~own)

    if popcount(bb[BISHOP]) >= 2:
        feat[I_BISHOP_PAIR] += 1
    if popcount(bb[6 + BISHOP]) >= 2:
        feat[I_BISHOP_PAIR] -= 1

    feat[I_PAWN_ISLANDS] = _islands(wp, ev.file_bb) - _islands(bp, ev.file_bb)
    feat[I_KING_SHELTER] = popcount(wp & ev.shelter[WHITE, wking]) - popcount(
        bp & ev.shelter[BLACK, bking]
    )
    feat[I_KING_STORM] = popcount(bp & ev.passed[WHITE, wking]) - popcount(
        wp & ev.passed[BLACK, bking]
    )
    danger_w = units_w * (units_w + 1) // 2
    danger_b = units_b * (units_b + 1) // 2
    feat[I_KING_ATTACK] = danger_b - danger_w
    return mg, eg, phase


@njit
def _taper(mg: int, eg: int, phase: int) -> int:
    if phase > MAX_PHASE:
        phase = MAX_PHASE
    blend = mg * phase + eg * (MAX_PHASE - phase)
    return blend // MAX_PHASE if blend >= 0 else -(-blend // MAX_PHASE)


@njit
def evaluate(pos: Position, t: Tables, ev: EvalTables) -> int:
    """Static evaluation in centipawns from the side to move's point of view."""
    feat = ev.scratch
    mg, eg, phase = eval_components(pos, t, ev, feat)
    extra_mg = 0
    extra_eg = 0
    for i in range(N_TERMS):
        if i == I_TEMPO:
            continue
        extra_mg += ev.w_mg[i] * feat[i]
        extra_eg += ev.w_eg[i] * feat[i]
    score = _taper(mg + extra_mg, eg + extra_eg, phase)
    if pos.st[ST_SIDE] != WHITE:
        score = -score
    return score + _taper(int(ev.w_mg[I_TEMPO]), int(ev.w_eg[I_TEMPO]), phase)


@njit
def insufficient_material(pos: Position) -> bool:
    """King versus king with at most one minor piece on the board (a dead draw)."""
    bb = pos.bb
    heavy = bb[PAWN] | bb[ROOK] | bb[QUEEN] | bb[6 + PAWN] | bb[6 + ROOK] | bb[6 + QUEEN]
    if heavy:
        return False
    minors = bb[KNIGHT] | bb[BISHOP] | bb[6 + KNIGHT] | bb[6 + BISHOP]
    return popcount(minors) <= 1
