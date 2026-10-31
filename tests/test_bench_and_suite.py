from __future__ import annotations

from pathlib import Path

import chess
import chess.engine
import pytest

from chessbot.tools import bench, suite
from chessbot.tools.common import data_path
from chessbot.tools.epd import read_test_suite

SAMPLE_SUITE = """\
# mate in one for White, a free queen for Black, and an avoid-move position
6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - bm Ra8#; id "mate.001";
4k3/8/8/3q4/8/8/8/R3K2Q b - - bm Qxh1; id "grab.001";
4k3/8/8/3q4/8/8/8/R3K2Q b - - am Qa2; id "avoid.001";
"""


@pytest.fixture
def sample_suite(tmp_path: Path) -> Path:
    path = tmp_path / "sample.epd"
    path.write_text(SAMPLE_SUITE)
    return path


def test_bench_positions_are_legal() -> None:
    fens = bench.read_fens(data_path("bench.epd"))
    assert len(fens) >= 20
    for fen in fens:
        board = chess.Board(fen)
        assert board.is_valid()
        assert not board.is_game_over()


def test_bench_with_fixture_engine(greedy_engine: str) -> None:
    fens = bench.read_fens(data_path("bench.epd"))[:5]
    result = bench.run_bench(greedy_engine, depth=1, fens=fens)
    assert result.engine == "GreedyFixture"
    assert result.nodes == sum(chess.Board(f).legal_moves.count() for f in fens)


@pytest.mark.stockfish
def test_bench_is_deterministic(stockfish: str) -> None:
    fens = bench.read_fens(data_path("bench.epd"))[:6]
    first = bench.run_bench(stockfish, 8, fens)
    second = bench.run_bench(stockfish, 8, fens)
    assert first.nodes == second.nodes > 0


def test_read_test_suite(sample_suite: Path) -> None:
    entries = read_test_suite(sample_suite)
    assert [e.id for e in entries] == ["mate.001", "grab.001", "avoid.001"]
    assert entries[0].best == (chess.Move.from_uci("a1a8"),)
    assert entries[2].avoid == (chess.Move.from_uci("d5a2"),)


def test_scoring_rules(sample_suite: Path) -> None:
    mate, grab, avoid = read_test_suite(sample_suite)
    assert suite.score(mate, chess.Move.from_uci("a1a8")).solved
    assert not suite.score(mate, chess.Move.from_uci("a1a7")).solved
    assert not suite.score(grab, None).solved
    assert not suite.score(avoid, chess.Move.from_uci("d5a2")).solved
    assert suite.score(avoid, chess.Move.from_uci("d5h1")).solved


def test_sts_points() -> None:
    (entry,) = read_test_suite(Path(__file__).parent / "fixtures" / "sts_sample.epd")
    best = suite.score(entry, chess.Move.from_uci("f4f5"))
    assert (best.solved, best.points, best.max_points) == (True, 10, 10)
    partial = suite.score(entry, chess.Move.from_uci("d4f2"))
    assert (partial.solved, partial.points) == (False, 3)


def test_run_suite_with_fixture_engine(greedy_engine: str, sample_suite: Path) -> None:
    entries = read_test_suite(sample_suite)
    outcomes = suite.run_suite(greedy_engine, entries, chess.engine.Limit(depth=1), concurrency=2)
    assert [o.id for o in outcomes] == ["mate.001", "grab.001", "avoid.001"]
    assert [o.solved for o in outcomes] == [False, True, True]


@pytest.mark.stockfish
def test_suite_cli_with_stockfish(
    stockfish: str, sample_suite: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert suite.main([str(sample_suite), "--engine", stockfish, "--depth", "6"]) == 0
    assert "solved 3/3" in capsys.readouterr().out
