from __future__ import annotations

import threading
import time
from typing import Any

import chess
import numpy as np
import pytest

from chessbot.core import Board
from chessbot.core.position import START_FEN, move_to_uci
from chessbot.search import Limits, Searcher, SearchInfo, warmup
from chessbot.search.evaluate import eval_tables, evaluate, insufficient_material
from chessbot.search.negamax import MATE, is_repetition
from chessbot.search.searcher import plan_time
from chessbot.search.see import SEE_BUFFER, see
from chessbot.search.tt import (
    BOUND_LOWER,
    BOUND_UPPER,
    ENTRY_BYTES,
    clear_tt,
    new_tt,
    score_from_tt,
    score_to_tt,
    tt_index,
    tt_store,
)

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
EVAL_FENS = [
    START_FEN,
    KIWIPETE,
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def ev(fen: str) -> int:
    return int(evaluate(Board(fen).pos, eval_tables()))


def search(fen: str, **limits: Any) -> tuple[str, int, Board]:
    board = Board(fen)
    result = Searcher().search(board, Limits(**limits))
    return move_to_uci(result.best_move), result.score, board


def test_start_position_is_balanced() -> None:
    assert ev(START_FEN) == 0


@pytest.mark.parametrize("fen", EVAL_FENS)
def test_eval_is_colour_symmetric(fen: str) -> None:
    mirrored = chess.Board(fen).mirror().fen()
    assert ev(fen) == ev(mirrored)


def test_eval_is_from_the_side_to_moves_view() -> None:
    white_up_queen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert ev(white_up_queen) > 900
    assert ev(white_up_queen.replace(" w ", " b ")) == -ev(white_up_queen)


def test_eval_tapers_towards_the_endgame_tables() -> None:
    # A lone advanced pawn is worth far more in the endgame table (178 + 94) than in the
    # middlegame one (98 + 82); with no other material the phase is 0, so the endgame value wins.
    assert ev("4k3/P7/8/8/8/8/8/4K3 w - - 0 1") - ev("4k3/8/8/8/8/8/8/4K3 w - - 0 1") > 250


@pytest.mark.parametrize(
    ("fen", "draw"),
    [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", True),
        ("4k3/8/8/8/8/8/8/2N1K3 w - - 0 1", True),
        ("4k3/8/8/2b5/8/8/8/4K3 w - - 0 1", True),
        ("4k3/8/8/2b5/8/8/8/2N1K3 w - - 0 1", False),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", False),
    ],
)
def test_insufficient_material(fen: str, draw: bool) -> None:
    assert bool(insufficient_material(Board(fen).pos)) is draw


def test_repetition_detection() -> None:
    board = Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        assert not is_repetition(board.pos)
        board.push_uci(uci)
    assert is_repetition(board.pos)


@pytest.mark.parametrize(
    ("fen", "best"),
    [
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    ],
)
def test_finds_mate_in_one(fen: str, best: str) -> None:
    move, score, _ = search(fen, depth=3)
    assert move == best
    assert score == MATE - 1


def test_finds_mate_in_two() -> None:
    move, score, _ = search("kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1", depth=4)
    assert move == "a1a6"
    assert score == MATE - 3


def test_wins_a_hanging_queen() -> None:
    move, score, _ = search("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", depth=3)
    assert move == "d2d5"
    assert score > 300


def test_prefers_mate_to_stalemate() -> None:
    # Qb6 and Qc7 stalemate the bare king; Qc8 mates.
    move, score, _ = search("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1", depth=4)
    assert move == "c1c8"
    assert score == MATE - 1


def test_stalemate_scores_zero() -> None:
    _, score, _ = search("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1", depth=2)
    assert score == 0


def test_fifty_move_rule_saves_the_losing_side() -> None:
    lost = "k7/8/8/8/8/8/5q2/K7 w - - 0 80"
    _, score, _ = search(lost, depth=3)
    assert score < -500
    _, score, _ = search(lost.replace(" 0 80", " 99 80"), depth=3)
    assert score == 0


def test_no_legal_moves_returns_null_move() -> None:
    board = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert Searcher().search(board, Limits(depth=3)).best_move == 0


def test_search_restores_the_position() -> None:
    board = Board(KIWIPETE)
    key = board.key
    Searcher().search(board, Limits(depth=3))
    assert board.fen() == KIWIPETE
    assert board.key == key
    Searcher().search(board, Limits(nodes=5000))  # abandoned mid-iteration
    assert board.fen() == KIWIPETE
    assert board.key == key


def test_node_limit_is_exact_and_deterministic() -> None:
    a = Searcher().search(Board(KIWIPETE), Limits(nodes=12345))
    b = Searcher().search(Board(KIWIPETE), Limits(nodes=12345))
    assert a.nodes == b.nodes == 12345
    assert a.best_move == b.best_move
    assert Searcher().search(Board(KIWIPETE), Limits(depth=4)).nodes == (
        Searcher().search(Board(KIWIPETE), Limits(depth=4)).nodes
    )


def test_principal_variation_is_legal_and_deepens() -> None:
    infos: list[SearchInfo] = []
    Searcher().search(Board(KIWIPETE), Limits(depth=4), on_info=infos.append)
    assert [i.depth for i in infos] == [1, 2, 3, 4]
    for info in infos:
        board = chess.Board(KIWIPETE)
        for m in info.pv:
            move = chess.Move.from_uci(move_to_uci(m))
            assert move in board.legal_moves
            board.push(move)
        assert len(info.pv) >= 1


def test_movetime_is_respected() -> None:
    warmup()
    start = time.perf_counter()
    result = Searcher().search(Board(KIWIPETE), Limits(movetime=300))
    elapsed = time.perf_counter() - start
    assert result.best_move
    assert elapsed < 0.6


def test_stop_event_ends_an_infinite_search() -> None:
    stop = threading.Event()
    timer = threading.Timer(0.3, stop.set)
    timer.start()
    start = time.perf_counter()
    result = Searcher().search(Board(KIWIPETE), Limits(infinite=True), stop=stop)
    assert time.perf_counter() - start < 1.0
    assert result.best_move


def test_mate_reporting() -> None:
    assert SearchInfo(1, 1, MATE - 1, 0, 0.0).mate_in == 1
    assert SearchInfo(1, 1, MATE - 3, 0, 0.0).mate_in == 2
    assert SearchInfo(1, 1, -(MATE - 2), 0, 0.0).mate_in == -1
    assert SearchInfo(1, 1, 150, 0, 0.0).mate_in is None


def test_time_plan() -> None:
    budget = plan_time(Limits(wtime=10_000, btime=5_000, winc=100), True, 50)
    assert budget.soft_ms is not None
    assert budget.hard_ms is not None
    assert 300 < budget.soft_ms < 500
    assert budget.soft_ms <= budget.hard_ms <= 5_000
    black = plan_time(Limits(wtime=10_000, btime=5_000), False, 50)
    assert black.soft_ms is not None
    assert black.soft_ms < budget.soft_ms
    assert plan_time(Limits(movetime=1000), True, 50).hard_ms == 950
    assert plan_time(Limits(infinite=True, wtime=1000), True, 50).hard_ms is None
    assert plan_time(Limits(depth=5), True, 50).hard_ms is None
    tiny = plan_time(Limits(wtime=30, btime=30), True, 50)
    assert tiny.hard_ms is not None
    assert tiny.hard_ms >= 1
    assert plan_time(Limits(wtime=60_000, movestogo=2), True, 50).hard_ms is not None


def test_tt_mate_scores_are_stored_relative_to_the_node() -> None:
    mate_in_3_from_ply_5 = MATE - 10
    stored = score_to_tt(mate_in_3_from_ply_5, 5)
    assert stored == MATE - 5
    assert score_from_tt(stored, 2) == MATE - 7
    assert score_from_tt(score_to_tt(-(MATE - 9), 4), 4) == -(MATE - 9)
    assert score_to_tt(123, 7) == 123


def test_tt_size_and_store() -> None:
    tt = new_tt(1)
    assert tt.keys.shape[0] == (1 << 20) // ENTRY_BYTES
    key = np.uint64(0x1234_5678_9ABC_DEF1)
    tt_store(tt, key, 777, 50, 0, 6, BOUND_LOWER, 0)
    i = tt_index(tt, key)
    assert (tt.keys[i], tt.moves[i], tt.scores[i], tt.depths[i]) == (key, 777, 50, 6)
    # A shallower bound for the same position keeps the deeper entry...
    tt_store(tt, key, 0, 10, 0, 2, BOUND_UPPER, 0)
    assert tt.depths[i] == 6
    # ...but a different position always takes the slot.
    other = key ^ np.uint64(1 << 40)
    tt_store(tt, other, 0, -5, 0, 1, BOUND_UPPER, 0)
    assert tt.keys[i] == other
    clear_tt(tt)
    assert not tt.keys.any()


def test_new_game_clears_the_table() -> None:
    searcher = Searcher(hash_mb=1)
    searcher.search(Board(KIWIPETE), Limits(depth=4))
    assert searcher.tt.keys.any()
    searcher.new_game()
    assert not searcher.tt.keys.any()


def test_table_reuse_saves_nodes_on_the_next_search() -> None:
    searcher = Searcher(hash_mb=1)
    first = searcher.search(Board(KIWIPETE), Limits(depth=5))
    again = searcher.search(Board(KIWIPETE), Limits(depth=5))
    assert again.best_move == first.best_move
    assert again.nodes < first.nodes


@pytest.mark.parametrize(
    ("fen", "uci", "value"),
    [
        ("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5", 100),
        ("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", -220),
        ("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0),
        ("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1", "d2d5", -800),
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q", 800),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100),
    ],
)
def test_static_exchange_evaluation(fen: str, uci: str, value: int) -> None:
    board = Board(fen)
    move = next(m for m in board.legal_moves() if move_to_uci(m) == uci)
    assert see(board.pos, board.t, move, np.zeros(SEE_BUFFER, np.int32)) == value
