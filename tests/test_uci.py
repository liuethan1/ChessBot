from __future__ import annotations

import io
import shlex
import subprocess
import sys
import time

import chess
import chess.engine
import pytest

from chessbot.search import SearchInfo
from chessbot.search.negamax import MATE
from chessbot.tools.bench import run_bench
from chessbot.tools.perft import UciBackend
from chessbot.uci import UciEngine, format_info, main, parse_go

ENGINE = [sys.executable, "-m", "chessbot"]


def run_commands(*lines: str) -> list[str]:
    out = io.StringIO()
    engine = UciEngine(out)
    for line in lines:
        engine.handle(line)
    engine.wait_search()
    return out.getvalue().splitlines()


def test_handshake_and_options() -> None:
    lines = run_commands("uci", "isready")
    assert lines[0].startswith("id name ChessBot")
    assert "uciok" in lines
    assert lines[-1] == "readyok"
    assert any("UCI_Elo" in line for line in lines)
    assert any("Move Overhead" in line for line in lines)
    assert any(line.startswith("option name Hash type spin") for line in lines)


def test_hash_option_resizes_the_table() -> None:
    out = io.StringIO()
    engine = UciEngine(out)
    for line in ("isready", "setoption name Hash value 2"):
        engine.handle(line)
    assert engine.searcher is not None
    assert engine.searcher.tt.keys.nbytes == (2 << 20) // 2


def test_command_line_runs_one_command() -> None:
    out = io.StringIO()
    assert main(stdin=io.StringIO("go depth 1\n"), stdout=out, argv=["bench", "2"]) == 0
    lines = out.getvalue().splitlines()
    assert any(line.startswith("Nodes searched") for line in lines)
    assert not any(line.startswith("bestmove") for line in lines)


def test_bench_command_is_deterministic() -> None:
    first = run_commands("isready", "bench 3")
    second = run_commands("isready", "bench 3")
    nodes = [line for line in first if line.startswith("Nodes searched")]
    assert len(nodes) == 1
    assert int(nodes[0].split(":")[1]) > 0
    assert nodes == [line for line in second if line.startswith("Nodes searched")]


def test_parse_go() -> None:
    limits = parse_go(
        ["wtime", "1000", "btime", "2000", "winc", "10", "binc", "20", "movestogo", "5"]
    )
    assert (limits.wtime, limits.btime, limits.winc, limits.binc) == (1000, 2000, 10, 20)
    assert limits.movestogo == 5
    assert parse_go(["infinite"]).infinite
    assert parse_go(["depth", "7"]).depth == 7
    assert parse_go(["nodes", "100", "movetime", "50"]).nodes == 100


def test_go_depth_plays_a_legal_move_after_moves() -> None:
    lines = run_commands("isready", "position startpos moves e2e4 e7e5 g1f3", "go depth 3")
    best = lines[-1].split()
    assert best[0] == "bestmove"
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
    assert chess.Move.from_uci(best[1]) in board.legal_moves
    assert any(line.startswith("info depth 3 ") for line in lines)


def test_position_fen_and_setoption() -> None:
    out = io.StringIO()
    engine = UciEngine(out)
    fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    engine.handle(f"position fen {fen}")
    engine.handle("setoption name Move Overhead value 120")
    engine.handle("setoption name UCI_LimitStrength value true")
    engine.handle("setoption name UCI_Elo value 99999")
    engine.handle("d")
    assert (engine.move_overhead, engine.limit_strength, engine.elo) == (120, True, 3190)
    engine.handle("go depth 2")
    engine.wait_search()
    lines = out.getvalue().splitlines()
    assert f"Fen: {fen}" in lines
    assert "score mate 1" in lines[-2]
    assert lines[-1] == "bestmove d1d8"


def test_stop_ends_infinite_search() -> None:
    out = io.StringIO()
    engine = UciEngine(out)
    engine.handle("isready")
    engine.handle("go infinite")
    time.sleep(0.3)
    assert "bestmove" not in out.getvalue()
    engine.handle("stop")
    assert out.getvalue().splitlines()[-1].startswith("bestmove ")


def test_mated_position_answers_null_move() -> None:
    lines = run_commands("position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "go depth 3")
    assert lines[-1] == "bestmove 0000"


def test_go_perft() -> None:
    lines = run_commands("position startpos", "go perft 3")
    assert lines[-1] == "Nodes searched: 8902"
    assert "e2e4: 600" in lines


def test_format_info_mate_and_cp() -> None:
    assert "score mate 2 " in format_info(SearchInfo(4, 6, MATE - 3, 100, 0.01, []))
    assert "score cp -15 " in format_info(SearchInfo(4, 6, -15, 100, 0.0, []))


def test_subprocess_engine_with_python_chess() -> None:
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    try:
        board = chess.Board()
        for _ in range(6):
            result = engine.play(board, chess.engine.Limit(time=0.05))
            assert result.move is not None
            assert result.move in board.legal_moves
            board.push(result.move)
        info = engine.analyse(chess.Board(), chess.engine.Limit(depth=4))
        assert info["depth"] == 4
        assert info["pv"]
    finally:
        engine.quit()


def test_quit_exits_promptly() -> None:
    proc = subprocess.run(
        ENGINE,
        input="uci\nisready\ngo infinite\nquit\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "readyok" in proc.stdout


def test_perft_tool_uci_backend() -> None:
    backend = UciBackend(shlex.join(ENGINE))
    try:
        assert backend.perft(chess.STARTING_FEN, 3) == 8902
        assert backend.divide(chess.STARTING_FEN, 2)["e2e4"] == 20
    finally:
        backend.close()


@pytest.mark.slow
def test_bench_tool_runs_the_engine() -> None:
    result = run_bench(shlex.join(ENGINE), 3, [chess.STARTING_FEN, chess.Board().fen()])
    assert result.engine.startswith("ChessBot")
    assert result.positions[0].nodes == result.positions[1].nodes > 0
