from __future__ import annotations

import io
from pathlib import Path

import chess
import chess.pgn
import pytest

from chessbot.tools import acpl


def _game(pgn: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return game


def test_losses_from_scripted_evals() -> None:
    # Evals from the side to move, one per position: start, after e4, after e5, after Qh5.
    evals = iter([30, -30, 200, -900])
    report = acpl.analyse_game(_game("1. e4 e5 2. Qh5"), lambda _: next(evals))
    assert report.white.losses == [0, 0]
    assert report.black.losses == [170]
    assert report.black.mistakes == 1
    assert report.white.acpl == 0


def test_clamping_and_loss_cap() -> None:
    evals = iter([0, 5000, -5000])
    report = acpl.analyse_game(_game("1. e4 e5"), lambda _: next(evals), cap=1000, max_loss=600)
    assert report.white.losses == [600]
    assert report.black.losses == [0]
    assert report.white.blunders == 1


def test_checkmate_is_scored_without_engine() -> None:
    calls = []

    def evaluate(board: chess.Board) -> int:
        calls.append(board.fen())
        return 0

    report = acpl.analyse_game(_game("1. f3 e5 2. g4 Qh4#"), evaluate)
    assert len(calls) == 4
    assert report.black.losses == [0, 0]


@pytest.mark.slow
@pytest.mark.stockfish
def test_reproduces_legacy_baseline(stockfish: str, legacy_game: Path) -> None:
    evaluator = acpl.StockfishEvaluator(stockfish, depth=14)
    try:
        (report,) = acpl.analyse_pgn(legacy_game, evaluator)
    finally:
        evaluator.close()
    assert (report.white.moves, report.black.moves) == (83, 82)
    assert 320 <= acpl.combined_acpl(report) <= 400
    assert report.white.blunders >= 30
    assert report.black.blunders >= 30
