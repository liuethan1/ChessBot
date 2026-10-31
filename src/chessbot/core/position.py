"""Position state, move encoding, FEN I/O, Zobrist keys, and make/unmake.

A :class:`Position` is a named tuple of numpy arrays so it can be passed straight into jitted
code. ``make_move`` pushes the information needed to undo a move onto ``undo`` and stores the new
Zobrist key in ``keys[ply]``, so ``keys[: ply + 1]`` is the game's key history (for repetition
detection later).

Moves are ``int`` values: ``from | to << 6 | flag << 12``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import (
    BLACK,
    KING,
    NO_PIECE,
    NO_SQ,
    PAWN,
    ROOK,
    WHITE,
    WHITE_OCC,
    ZERO,
    ZOB_CASTLE,
    ZOB_EP,
    ZOB_SIDE,
    Tables,
    bit,
)
from chessbot.core.jit import njit

QUIET, DOUBLE_PUSH, KING_CASTLE, QUEEN_CASTLE, CAPTURE, EP_CAPTURE = 0, 1, 2, 3, 4, 5
PROMO = 8
"""Promotion flags are ``PROMO | (piece - KNIGHT)``, plus ``CAPTURE`` for promotion captures."""
PROMO_CAPTURE = PROMO | CAPTURE

ST_SIDE, ST_CASTLE, ST_EP, ST_HALFMOVE, ST_FULLMOVE, ST_PLY = range(6)
U_MOVE, U_CAPTURED, U_CASTLE, U_EP, U_HALFMOVE = range(5)
MAX_PLY = 4096
"""Capacity of the make/unmake stack (game plies plus search depth)."""

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PIECE_CHARS = "PNBRQKpnbrqk"


class Position(NamedTuple):
    bb: npt.NDArray[np.uint64]
    """(14,): one bitboard per piece code, then white and black occupancy."""
    board: npt.NDArray[np.int8]
    """(64,): piece code on each square, ``NO_PIECE`` if empty."""
    st: npt.NDArray[np.int64]
    """Side to move, castling rights (K=1, Q=2, k=4, q=8), en passant square, clocks, ply."""
    keys: npt.NDArray[np.uint64]
    undo: npt.NDArray[np.int32]


@njit
def encode_move(frm: int, to: int, flag: int) -> int:
    return frm | (to << 6) | (flag << 12)


@njit
def move_from(m: int) -> int:
    return m & 63


@njit
def move_to(m: int) -> int:
    return (m >> 6) & 63


@njit
def move_flag(m: int) -> int:
    return (m >> 12) & 15


@njit
def _put(pos: Position, pc: int, sq: int) -> None:
    b = bit(sq)
    pos.bb[pc] |= b
    pos.bb[WHITE_OCC + pc // 6] |= b
    pos.board[sq] = pc


@njit
def _remove(pos: Position, pc: int, sq: int) -> None:
    b = bit(sq)
    pos.bb[pc] ^= b
    pos.bb[WHITE_OCC + pc // 6] ^= b
    pos.board[sq] = NO_PIECE


@njit
def compute_key(pos: Position, t: Tables) -> np.uint64:
    """Zobrist key from scratch (``make_move`` updates it incrementally)."""
    key = ZERO
    for sq in range(64):
        pc = pos.board[sq]
        if pc != NO_PIECE:
            key ^= t.zobrist[pc * 64 + sq]
    key ^= t.zobrist[ZOB_CASTLE + pos.st[ST_CASTLE]]
    if pos.st[ST_EP] != NO_SQ:
        key ^= t.zobrist[ZOB_EP + (pos.st[ST_EP] & 7)]
    if pos.st[ST_SIDE] == BLACK:
        key ^= t.zobrist[ZOB_SIDE]
    return key  # type: ignore[no-any-return]


@njit
def _ep_capturable(pos: Position, t: Tables, ep: int, capturer: int) -> bool:
    """Whether a ``capturer`` pawn attacks ``ep``. Only then is the square part of the key."""
    return bool(t.leaper[capturer ^ 1, ep] & pos.bb[capturer * 6 + PAWN])


@njit
def make_move(pos: Position, t: Tables, m: int) -> None:
    board, st = pos.board, pos.st
    ply = st[ST_PLY]
    frm, to, flag = m & 63, (m >> 6) & 63, (m >> 12) & 15
    us = st[ST_SIDE]
    them = us ^ 1
    pc = int(board[frm])
    key = pos.keys[ply]

    u = pos.undo[ply]
    u[U_MOVE] = m
    u[U_CASTLE] = st[ST_CASTLE]
    u[U_EP] = st[ST_EP]
    u[U_HALFMOVE] = st[ST_HALFMOVE]

    if st[ST_EP] != NO_SQ:
        key ^= t.zobrist[ZOB_EP + (st[ST_EP] & 7)]
        st[ST_EP] = NO_SQ

    captured = NO_PIECE
    if flag == EP_CAPTURE:
        capsq = to - 8 if us == WHITE else to + 8
        captured = them * 6 + PAWN
        _remove(pos, captured, capsq)
        key ^= t.zobrist[captured * 64 + capsq]
    elif flag & CAPTURE:
        captured = int(board[to])
        _remove(pos, captured, to)
        key ^= t.zobrist[captured * 64 + to]
    u[U_CAPTURED] = captured

    _remove(pos, pc, frm)
    key ^= t.zobrist[pc * 64 + frm]
    reset_clock = pc % 6 == PAWN or captured != NO_PIECE
    if flag & PROMO:
        pc = us * 6 + (flag & 3) + 1
    _put(pos, pc, to)
    key ^= t.zobrist[pc * 64 + to]

    if flag in (KING_CASTLE, QUEEN_CASTLE):
        rook = us * 6 + ROOK
        rfrom, rto = (frm + 3, frm + 1) if flag == KING_CASTLE else (frm - 4, frm - 1)
        _remove(pos, rook, rfrom)
        _put(pos, rook, rto)
        key ^= t.zobrist[rook * 64 + rfrom] ^ t.zobrist[rook * 64 + rto]

    castle = st[ST_CASTLE]
    if castle:
        new_castle = castle & t.castle_mask[frm] & t.castle_mask[to]
        if new_castle != castle:
            key ^= t.zobrist[ZOB_CASTLE + castle] ^ t.zobrist[ZOB_CASTLE + new_castle]
            st[ST_CASTLE] = new_castle

    if flag == DOUBLE_PUSH:
        ep = (frm + to) >> 1
        if _ep_capturable(pos, t, ep, them):
            st[ST_EP] = ep
            key ^= t.zobrist[ZOB_EP + (ep & 7)]

    if reset_clock:
        st[ST_HALFMOVE] = 0
    else:
        st[ST_HALFMOVE] += 1
    if us == BLACK:
        st[ST_FULLMOVE] += 1
    st[ST_SIDE] = them
    key ^= t.zobrist[ZOB_SIDE]
    st[ST_PLY] = ply + 1
    pos.keys[ply + 1] = key


@njit
def unmake_move(pos: Position, t: Tables) -> None:
    st = pos.st
    ply = st[ST_PLY] - 1
    st[ST_PLY] = ply
    u = pos.undo[ply]
    m = int(u[U_MOVE])
    frm, to, flag = m & 63, (m >> 6) & 63, (m >> 12) & 15
    st[ST_CASTLE] = u[U_CASTLE]
    st[ST_EP] = u[U_EP]
    st[ST_HALFMOVE] = u[U_HALFMOVE]
    us = st[ST_SIDE] ^ 1
    st[ST_SIDE] = us
    if us == BLACK:
        st[ST_FULLMOVE] -= 1

    pc = int(pos.board[to])
    _remove(pos, pc, to)
    if flag & PROMO:
        pc = us * 6 + PAWN
    _put(pos, pc, frm)

    captured = int(u[U_CAPTURED])
    if flag == EP_CAPTURE:
        _put(pos, captured, to - 8 if us == WHITE else to + 8)
    elif captured != NO_PIECE:
        _put(pos, captured, to)

    if flag in (KING_CASTLE, QUEEN_CASTLE):
        rook = us * 6 + ROOK
        rfrom, rto = (frm + 3, frm + 1) if flag == KING_CASTLE else (frm - 4, frm - 1)
        _remove(pos, rook, rto)
        _put(pos, rook, rfrom)


@njit
def make_null_move(pos: Position, t: Tables) -> None:
    st = pos.st
    ply = st[ST_PLY]
    key = pos.keys[ply]
    u = pos.undo[ply]
    u[U_MOVE] = 0
    u[U_CAPTURED] = NO_PIECE
    u[U_CASTLE] = st[ST_CASTLE]
    u[U_EP] = st[ST_EP]
    u[U_HALFMOVE] = st[ST_HALFMOVE]
    if st[ST_EP] != NO_SQ:
        key ^= t.zobrist[ZOB_EP + (st[ST_EP] & 7)]
        st[ST_EP] = NO_SQ
    st[ST_HALFMOVE] += 1
    st[ST_SIDE] ^= 1
    key ^= t.zobrist[ZOB_SIDE]
    st[ST_PLY] = ply + 1
    pos.keys[ply + 1] = key


@njit
def unmake_null_move(pos: Position, t: Tables) -> None:
    st = pos.st
    ply = st[ST_PLY] - 1
    st[ST_PLY] = ply
    u = pos.undo[ply]
    st[ST_EP] = u[U_EP]
    st[ST_HALFMOVE] = u[U_HALFMOVE]
    st[ST_SIDE] ^= 1


def empty_position() -> Position:
    return Position(
        np.zeros(14, np.uint64),
        np.full(64, NO_PIECE, np.int8),
        np.zeros(8, np.int64),
        np.zeros(MAX_PLY + 1, np.uint64),
        np.zeros((MAX_PLY, 5), np.int32),
    )


def square_name(sq: int) -> str:
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"bad square {name!r}")
    return "abcdefgh".index(name[0]) + 8 * (int(name[1]) - 1)


def from_fen(fen: str, t: Tables) -> Position:
    """Parse a FEN. The clocks are optional; castling rights without the king and rook on their
    original squares, and en passant squares no pawn can capture on, are dropped."""
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"FEN needs at least 4 fields: {fen!r}")
    pos = empty_position()
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN board needs 8 ranks: {fen!r}")
    for i, rank in enumerate(ranks):
        f = 0
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            elif ch in PIECE_CHARS and f < 8:
                _put(pos, PIECE_CHARS.index(ch), (7 - i) * 8 + f)
                f += 1
            else:
                raise ValueError(f"bad FEN board: {fen!r}")
        if f != 8:
            raise ValueError(f"bad FEN rank {rank!r}")
    for color in (WHITE, BLACK):
        if bin(int(pos.bb[color * 6 + KING])).count("1") != 1:
            raise ValueError(f"each side needs exactly one king: {fen!r}")
    if fields[1] not in ("w", "b"):
        raise ValueError(f"bad side to move: {fen!r}")
    st = pos.st
    st[ST_SIDE] = WHITE if fields[1] == "w" else BLACK

    castle = 0
    board = pos.board
    for ch, right, ksq, rsq, color in (
        ("K", 1, 4, 7, WHITE),
        ("Q", 2, 4, 0, WHITE),
        ("k", 4, 60, 63, BLACK),
        ("q", 8, 60, 56, BLACK),
    ):
        if ch in fields[2] and board[ksq] == color * 6 + KING and board[rsq] == color * 6 + ROOK:
            castle |= right
    st[ST_CASTLE] = castle

    st[ST_EP] = NO_SQ
    if fields[3] != "-":
        ep = parse_square(fields[3])
        if _ep_capturable(pos, t, ep, int(st[ST_SIDE])):
            st[ST_EP] = ep
    st[ST_HALFMOVE] = int(fields[4]) if len(fields) > 4 else 0
    st[ST_FULLMOVE] = int(fields[5]) if len(fields) > 5 else 1
    st[ST_PLY] = 0
    pos.keys[0] = compute_key(pos, t)
    return pos


def to_fen(pos: Position) -> str:
    rows = []
    for r in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            pc = int(pos.board[r * 8 + f])
            if pc == NO_PIECE:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += PIECE_CHARS[pc]
        rows.append(row + (str(empty) if empty else ""))
    st = pos.st
    castle = "".join(c for c, b in zip("KQkq", (1, 2, 4, 8), strict=True) if st[ST_CASTLE] & b)
    ep = square_name(int(st[ST_EP])) if st[ST_EP] != NO_SQ else "-"
    side = "w" if st[ST_SIDE] == WHITE else "b"
    clocks = f"{int(st[ST_HALFMOVE])} {int(st[ST_FULLMOVE])}"
    return f"{'/'.join(rows)} {side} {castle or '-'} {ep} {clocks}"


def move_to_uci(m: int) -> str:
    flag = (m >> 12) & 15
    s = square_name(m & 63) + square_name((m >> 6) & 63)
    if flag & PROMO:
        s += "nbrq"[flag & 3]
    return s
