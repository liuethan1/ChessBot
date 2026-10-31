from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chessbot.core import Board, perft_fen, warmup
from chessbot.core.bitboard import (
    BISHOP_IDX,
    ROOK_IDX,
    bishop_attacks,
    rook_attacks,
    slider_attacks_slow,
    tables,
)
from chessbot.core.position import (
    START_FEN,
    compute_key,
    make_null_move,
    move_to_uci,
    unmake_null_move,
)
from chessbot.tools.common import data_path
from chessbot.tools.epd import read_perft_suite
from chessbot.tools.perft import FunctionBackend, find_mismatch, run_suite
from chessbot.tools.warmup import main as warmup_main

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
SUITE_FENS = [
    e.fen
    for name in ("cpw.epd", "standard.epd")
    for e in read_perft_suite(data_path("perft", name))
]


def random_game_fens(games: int, plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens = []
    for start in [START_FEN, KIWIPETE, *SUITE_FENS[::8]][:games]:
        board = chess.Board(start)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            fens.append(board.fen())
    return fens


GAME_FENS = random_game_fens(games=20, plies=120, seed=7)


def test_magic_lookups_match_ray_walking() -> None:
    t = tables()
    rng = np.random.default_rng(3)
    for _ in range(5000):
        sq = int(rng.integers(64))
        occ = rng.integers(0, 2**64, dtype=np.uint64) & rng.integers(0, 2**64, dtype=np.uint64)
        assert rook_attacks(t, sq, occ) == slider_attacks_slow(sq, occ, ROOK_IDX)
        assert bishop_attacks(t, sq, occ) == slider_attacks_slow(sq, occ, BISHOP_IDX)


@pytest.mark.parametrize("fen", SUITE_FENS[:40] + GAME_FENS[::10])
def test_fen_round_trip_matches_python_chess(fen: str) -> None:
    assert Board(fen).fen() == chess.Board(fen).fen(en_passant="xfen")


def test_fen_drops_impossible_castling_and_en_passant() -> None:
    assert Board("4k3/8/8/8/8/8/8/R3K1R1 w KQkq e3 3 9").fen() == "4k3/8/8/8/8/8/8/R3K1R1 w Q - 3 9"
    assert Board("4k3/8/8/8/8/8/8/4K3 b - -").fen() == "4k3/8/8/8/8/8/8/4K3 b - - 0 1"


@pytest.mark.parametrize("fen", ["", "8/8/8/8/8/8/8/8 w - - 0 1", "rnbqkbnr/pppppppp w KQkq - 0 1"])
def test_bad_fen_raises(fen: str) -> None:
    with pytest.raises(ValueError, match=r"FEN|king"):
        Board(fen)


def test_legal_moves_match_python_chess() -> None:
    for fen in SUITE_FENS + GAME_FENS:
        ours = Board(fen).legal_uci()
        assert len(ours) == len(set(ours)), fen
        assert sorted(ours) == sorted(m.uci() for m in chess.Board(fen).legal_moves), fen


def test_tactical_moves_are_captures_and_promotions() -> None:
    for fen in SUITE_FENS + GAME_FENS:
        ref = chess.Board(fen)
        want = sorted(
            m.uci() for m in ref.legal_moves if ref.is_capture(m) or m.promotion is not None
        )
        board = Board(fen)
        assert sorted(move_to_uci(m) for m in board.legal_moves(tactical_only=True)) == want, fen


def test_in_check_matches_python_chess() -> None:
    for fen in GAME_FENS:
        assert Board(fen).in_check() == chess.Board(fen).is_check(), fen


def test_make_unmake_restores_state_and_keys_stay_consistent() -> None:
    rng = random.Random(11)
    for start in (START_FEN, KIWIPETE, *SUITE_FENS[::16]):
        board = Board(start)
        before = [a.copy() for a in board.pos]
        ref = chess.Board(start)
        for _ in range(150):
            moves = board.legal_moves()
            if not moves:
                break
            move = rng.choice(moves)
            ref.push_uci(move_to_uci(move))
            board.push(move)
            assert board.fen() == ref.fen(en_passant="xfen")
            assert board.key == int(compute_key(board.pos, board.t))
            assert board.key == Board(board.fen()).key
        while board.ply:
            board.pop()
        assert board.fen() == Board(start).fen()
        for after, orig in zip(board.pos[:3], before[:3], strict=True):
            assert np.array_equal(after, orig)
        assert board.pos.keys[0] == before[3][0]


def test_transposition_and_side_to_move_change_the_key() -> None:
    board = Board()
    start = board.key
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert board.key == start
    other = Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        other.push_uci(uci)
    direct = Board()
    for uci in ("g1f3", "e7e5", "e2e4"):
        direct.push_uci(uci)
    assert other.key == direct.key
    assert Board().key != Board(START_FEN.replace(" w ", " b ")).key


def test_null_move_round_trip() -> None:
    board = Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
    key = board.key
    make_null_move(board.pos, board.t)
    assert board.white_to_move
    assert board.key == int(compute_key(board.pos, board.t))
    unmake_null_move(board.pos, board.t)
    assert board.key == key
    assert board.fen() == "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"


def test_push_uci_rejects_illegal_moves() -> None:
    board = Board()
    with pytest.raises(ValueError, match="illegal move"):
        board.push_uci("e2e5")
    with pytest.raises(IndexError):
        board.pop()


def test_divide_matches_reference() -> None:
    board = Board(KIWIPETE)
    divide = board.divide(3)
    assert sum(divide.values()) == 97_862
    ref = chess.Board(KIWIPETE)
    assert sorted(divide) == sorted(m.uci() for m in ref.legal_moves)


def test_perft_suites_through_the_harness() -> None:
    backend = FunctionBackend("chessbot.core.perft:perft_fen")
    for suite in ("cpw.epd", "standard.epd"):
        entries = read_perft_suite(data_path("perft", suite))
        results = run_suite(backend, entries, max_depth=6, max_nodes=2_000_000, deepest_only=False)
        assert results
        assert all(r.ok for r in results), [r for r in results if not r.ok]
    assert find_mismatch(backend, KIWIPETE, 3) is None


def test_perft_edge_depths() -> None:
    assert perft_fen(START_FEN, 0) == 1
    assert perft_fen(START_FEN, 1) == 20
    assert perft_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1", 1) == 15


def test_warmup_and_warm_start_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert warmup() >= 0
    assert warmup_main([]) == 0
    assert warmup_main(["--check", "60"]) == 0
    assert "warm start" in capsys.readouterr().out
