"""Bitboard primitives, precomputed attack tables, and magic bitboards for sliding pieces.

Squares are numbered a1 = 0, b1 = 1, ..., h8 = 63. Every bitboard is a ``numpy.uint64``. Inside
jitted code, never mix a ``uint64`` with a plain Python ``int`` in arithmetic: numba promotes
``uint64 op int64`` to ``float64``. Use ``ONE``/``ZERO``/``FULL`` and cast shift counts with
``U64(...)``.

All tables live in a :class:`Tables` named tuple that is passed into every jitted function.
Global arrays would be frozen into the compiled code as constants, which makes compilation slow
and the on-disk cache large.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from chessbot.core.jit import njit

U64 = np.uint64
ZERO = U64(0)
ONE = U64(1)
FULL = U64(0xFFFF_FFFF_FFFF_FFFF)

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
NO_PIECE = 12
"""Piece codes are ``color * 6 + piece_type``; ``NO_PIECE`` marks an empty square."""
WHITE_OCC, BLACK_OCC = 12, 13
"""Indices of the per-colour occupancy bitboards in ``Position.bb``."""
NO_SQ = 64

PAWN_ATT_WHITE, PAWN_ATT_BLACK, KNIGHT_ATT, KING_ATT = range(4)
"""Rows of ``Tables.leaper``. Row ``color`` is the pawn-attack table for that colour."""
BISHOP_IDX, ROOK_IDX = 0, 1
"""Rows of the ``Tables.m*`` magic arrays."""

ZOB_CASTLE = 12 * 64
ZOB_EP = ZOB_CASTLE + 16
ZOB_SIDE = ZOB_EP + 8
ZOB_SIZE = ZOB_SIDE + 1

RANK_1 = U64(0xFF)
RANK_2 = U64(0xFF00)
RANK_7 = U64(0xFF << 48)
RANK_8 = U64(0xFF << 56)

MAGIC_SEED = 0x9E3779B97F4A7C15
ZOBRIST_SEED = 0x0C4E55B07

U64Array = npt.NDArray[np.uint64]
I64Array = npt.NDArray[np.int64]


@intrinsic
def _popcount(typingctx: Any, x: Any) -> Any:
    """Number of set bits (LLVM ``ctpop``)."""
    if x != types.uint64:
        return None

    def codegen(context: Any, builder: Any, sig: Any, args: Any) -> Any:
        return builder.ctpop(args[0])

    return types.int64(types.uint64), codegen


@intrinsic
def _lsb(typingctx: Any, x: Any) -> Any:
    """Index of the lowest set bit (LLVM ``cttz``). Undefined for 0."""
    if x != types.uint64:
        return None

    def codegen(context: Any, builder: Any, sig: Any, args: Any) -> Any:
        return builder.cttz(args[0], ir.Constant(ir.IntType(1), 1))

    return types.int64(types.uint64), codegen


popcount = cast(Callable[[np.uint64], int], _popcount)
lsb = cast(Callable[[np.uint64], int], _lsb)


@njit
def bit(sq: int) -> np.uint64:
    return ONE << U64(sq)


class Tables(NamedTuple):
    leaper: U64Array
    """(4, 64): white pawn, black pawn, knight, and king attacks."""
    mmask: U64Array
    """(2, 64) relevant-occupancy masks for bishops (row 0) and rooks (row 1)."""
    mmagic: U64Array
    mshift: U64Array
    moff: I64Array
    """(2, 64) offsets into ``mtable``."""
    mtable: U64Array
    between: U64Array
    """(64, 64) squares strictly between two aligned squares, else empty."""
    line: U64Array
    """(64, 64) the full line through two aligned squares, else empty."""
    zobrist: U64Array
    """Flat keys: piece * 64 + square, then castling (16), en passant file (8), side to move."""
    castle_mask: I64Array
    """Castling rights that survive a move touching each square."""


@njit
def bishop_attacks(t: Tables, sq: int, occ: np.uint64) -> np.uint64:
    idx = ((occ & t.mmask[0, sq]) * t.mmagic[0, sq]) >> t.mshift[0, sq]
    return t.mtable[t.moff[0, sq] + np.int64(idx)]  # type: ignore[no-any-return]


@njit
def rook_attacks(t: Tables, sq: int, occ: np.uint64) -> np.uint64:
    idx = ((occ & t.mmask[1, sq]) * t.mmagic[1, sq]) >> t.mshift[1, sq]
    return t.mtable[t.moff[1, sq] + np.int64(idx)]  # type: ignore[no-any-return]


@njit
def _on_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


@njit
def slider_attacks_slow(sq: int, occ: np.uint64, slider: int) -> np.uint64:
    """Ray-walking attacks, used only to build and test the magic tables."""
    if slider == BISHOP_IDX:
        dfs, drs = (1, 1, -1, -1), (1, -1, 1, -1)
    else:
        dfs, drs = (1, -1, 0, 0), (0, 0, 1, -1)
    att = ZERO
    for d in range(4):
        f, r = sq % 8 + dfs[d], sq // 8 + drs[d]
        while _on_board(f, r):
            b = bit(r * 8 + f)
            att |= b
            if occ & b:
                break
            f += dfs[d]
            r += drs[d]
    return att


@njit
def _relevant_mask(sq: int, slider: int) -> np.uint64:
    if slider == BISHOP_IDX:
        dfs, drs = (1, 1, -1, -1), (1, -1, 1, -1)
    else:
        dfs, drs = (1, -1, 0, 0), (0, 0, 1, -1)
    mask = ZERO
    for d in range(4):
        f, r = sq % 8 + dfs[d], sq // 8 + drs[d]
        while _on_board(f + dfs[d], r + drs[d]):
            mask |= bit(r * 8 + f)
            f += dfs[d]
            r += drs[d]
    return mask


@njit
def _xorshift(state: U64Array) -> np.uint64:
    x = state[0]
    x ^= x >> U64(12)
    x ^= x << U64(25)
    x ^= x >> U64(27)
    state[0] = x
    return x * U64(2685821657736338717)  # type: ignore[no-any-return]


@njit
def _find_magics(
    slider: int,
    mmask: U64Array,
    mmagic: U64Array,
    mshift: U64Array,
    moff: I64Array,
    mtable: U64Array,
    start: int,
    seed: np.uint64,
) -> int:
    """Search magics for every square with a fixed seed; returns the next free table offset."""
    state = np.empty(1, np.uint64)
    state[0] = seed
    occs = np.empty(4096, np.uint64)
    refs = np.empty(4096, np.uint64)
    used = np.empty(4096, np.uint64)
    epoch = np.zeros(4096, np.int64)
    attempt = 0
    off = start
    for sq in range(64):
        mask = _relevant_mask(sq, slider)
        bits = popcount(mask)
        n = 1 << bits
        sub = ZERO
        for i in range(n):
            occs[i] = sub
            refs[i] = slider_attacks_slow(sq, sub, slider)
            sub = (sub - mask) & mask
        shift = U64(64 - bits)
        while True:
            magic = _xorshift(state) & _xorshift(state) & _xorshift(state)
            if popcount((mask * magic) >> U64(56)) < 6:
                continue
            attempt += 1
            ok = True
            for i in range(n):
                idx = np.int64((occs[i] * magic) >> shift)
                if epoch[idx] != attempt:
                    epoch[idx] = attempt
                    used[idx] = refs[i]
                elif used[idx] != refs[i]:
                    ok = False
                    break
            if ok:
                break
        for i in range(n):
            mtable[off + i] = used[i] if epoch[i] == attempt else ZERO
        mmask[slider, sq] = mask
        mmagic[slider, sq] = magic
        mshift[slider, sq] = shift
        moff[slider, sq] = off
        off += n
    return off


@njit
def _leaper_attacks(sq: int, dfs: tuple[int, ...], drs: tuple[int, ...]) -> np.uint64:
    att = ZERO
    for d in range(len(dfs)):
        f, r = sq % 8 + dfs[d], sq // 8 + drs[d]
        if _on_board(f, r):
            att |= bit(r * 8 + f)
    return att


@njit
def _fill_leapers(leaper: U64Array) -> None:
    for sq in range(64):
        leaper[PAWN_ATT_WHITE, sq] = _leaper_attacks(sq, (-1, 1), (1, 1))
        leaper[PAWN_ATT_BLACK, sq] = _leaper_attacks(sq, (-1, 1), (-1, -1))
        leaper[KNIGHT_ATT, sq] = _leaper_attacks(
            sq, (1, 2, 2, 1, -1, -2, -2, -1), (2, 1, -1, -2, -2, -1, 1, 2)
        )
        leaper[KING_ATT, sq] = _leaper_attacks(
            sq, (1, 1, 0, -1, -1, -1, 0, 1), (0, 1, 1, 1, 0, -1, -1, -1)
        )


@njit
def _fill_lines(between: U64Array, line: U64Array) -> None:
    for a in range(64):
        for slider in range(2):
            ray_a = slider_attacks_slow(a, ZERO, slider)
            for b in range(64):
                if a == b or not (ray_a & bit(b)):
                    continue
                line[a, b] = (ray_a & slider_attacks_slow(b, ZERO, slider)) | bit(a) | bit(b)
                between[a, b] = slider_attacks_slow(a, bit(b), slider) & slider_attacks_slow(
                    b, bit(a), slider
                )


def _zobrist_keys() -> U64Array:
    rng = np.random.default_rng(ZOBRIST_SEED)
    keys: U64Array = rng.integers(0, 2**64, size=ZOB_SIZE, dtype=np.uint64, endpoint=False)
    return keys


def _castle_mask() -> I64Array:
    mask = np.full(64, 15, np.int64)
    mask[4] &= ~3  # e1
    mask[7] &= ~1  # h1
    mask[0] &= ~2  # a1
    mask[60] &= ~12  # e8
    mask[63] &= ~4  # h8
    mask[56] &= ~8  # a8
    return mask


def build_tables() -> Tables:
    leaper = np.zeros((4, 64), np.uint64)
    _fill_leapers(leaper)
    mmask = np.zeros((2, 64), np.uint64)
    mmagic = np.zeros((2, 64), np.uint64)
    mshift = np.zeros((2, 64), np.uint64)
    moff = np.zeros((2, 64), np.int64)
    mtable = np.zeros(5248 + 102400, np.uint64)
    off = _find_magics(BISHOP_IDX, mmask, mmagic, mshift, moff, mtable, 0, U64(MAGIC_SEED))
    off = _find_magics(ROOK_IDX, mmask, mmagic, mshift, moff, mtable, off, U64(MAGIC_SEED ^ 1))
    assert off == len(mtable)
    between = np.zeros((64, 64), np.uint64)
    line = np.zeros((64, 64), np.uint64)
    _fill_lines(between, line)
    return Tables(
        leaper, mmask, mmagic, mshift, moff, mtable, between, line, _zobrist_keys(), _castle_mask()
    )


_TABLES: Tables | None = None


def tables() -> Tables:
    """The process-wide attack tables, built on first use."""
    global _TABLES
    if _TABLES is None:
        _TABLES = build_tables()
    return _TABLES
