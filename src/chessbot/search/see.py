"""Static exchange evaluation: the material outcome of the capture sequence on one square."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from chessbot.core.bitboard import NO_PIECE, PAWN, QUEEN, WHITE, WHITE_OCC, Tables, bit, lsb
from chessbot.core.jit import njit
from chessbot.core.movegen import attackers_to
from chessbot.core.position import EP_CAPTURE, PROMO, Position

SEE_BUFFER = 40


@njit
def see_value(pt: int) -> int:
    if pt == 0:
        return 100
    if pt == 1:
        return 320
    if pt == 2:
        return 330
    if pt == 3:
        return 500
    if pt == 4:
        return 900
    return 20000


@njit
def see(pos: Position, t: Tables, m: int, gain: npt.NDArray[np.int32]) -> int:
    """Expected material gain of move ``m`` for the side making it, with both sides always
    recapturing with their least valuable attacker and free to stop. Pins are ignored.

    ``gain`` is scratch space of at least ``SEE_BUFFER`` entries.
    """
    frm, to, flag = m & 63, (m >> 6) & 63, (m >> 12) & 15
    bb, board = pos.bb, pos.board
    occ = bb[WHITE_OCC] | bb[WHITE_OCC + 1]
    mover = int(board[frm])
    side = mover // 6
    if flag == EP_CAPTURE:
        gain[0] = 100
        occ ^= bit(to - 8 if side == WHITE else to + 8)
    else:
        victim = int(board[to])
        gain[0] = see_value(victim % 6) if victim != NO_PIECE else 0
    on_square = see_value(mover % 6)
    if flag & PROMO:
        promoted = (flag & 3) + 1
        gain[0] += see_value(promoted) - see_value(PAWN)
        on_square = see_value(promoted)
    occ ^= bit(frm)
    d = 0
    side ^= 1
    while d < SEE_BUFFER - 2:
        d += 1
        gain[d] = on_square - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            break
        att = attackers_to(t, bb, to, occ, side) & occ
        if not att:
            break
        pt = 0
        for k in range(6):
            pt = k
            if att & bb[side * 6 + k]:
                break
        sq = lsb(att & bb[side * 6 + pt])
        occ ^= bit(sq)
        on_square = see_value(pt)
        if pt == PAWN and (to >> 3) in (0, 7):
            on_square = see_value(QUEEN)
        side ^= 1
    d -= 1
    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1
    return int(gain[0])
