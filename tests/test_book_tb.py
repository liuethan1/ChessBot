from __future__ import annotations

import io
import re
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy
import pytest

import chessbot
from chessbot.book import OpeningBook, default_book_path, encode_polyglot_move, game_ply
from chessbot.core.board import Board
from chessbot.core.position import move_to_uci
from chessbot.tablebase import Tablebases, piece_count
from chessbot.tools.book import write_book
from chessbot.uci import UciEngine


def run_commands(*lines: str) -> list[str]:
    out = io.StringIO()
    engine = UciEngine(out)
    for line in lines:
        engine.handle(line)
    engine.wait_search()
    return out.getvalue().splitlines()


FIXTURES = Path(__file__).parent / "fixtures"
SYZYGY_3 = FIXTURES / "syzygy"
NEEDS_TB = "3-piece Syzygy fixtures missing"

KQK_MATE = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
KRK_WIN = "8/8/8/8/8/8/4KR2/7k w - - 0 1"
KPK_WIN = "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"


def _write_book(path: Path, fen: str, moves: list[tuple[str, int]]) -> None:
    board = chess.Board(fen)
    counts: dict[int, dict[int, int]] = {chess.polyglot.zobrist_hash(board): {}}
    for uci, weight in moves:
        raw = encode_polyglot_move(board, chess.Move.from_uci(uci))
        counts[chess.polyglot.zobrist_hash(board)][raw] = weight
    write_book(counts, path, min_games=1)


def _push_engine_move(fen: str, move: int) -> chess.Board:
    root = chess.Board(fen)
    root.push(chess.Move.from_uci(move_to_uci(move)))
    return root


def test_game_ply_follows_fullmove_clock() -> None:
    assert game_ply(Board()) == 0
    board = Board()
    board.push_uci("e2e4")
    assert game_ply(board) == 1
    assert game_ply(Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")) == 1
    assert game_ply(Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 9")) == 17


def test_polyglot_castling_encodes_king_to_rook() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    raw = encode_polyglot_move(board, chess.Move.from_uci("e1g1"))
    assert raw & 63 == chess.H1
    assert (raw >> 6) & 63 == chess.E1


def test_packaged_book_covers_the_start_position() -> None:
    path = default_book_path()
    assert path.is_file()
    book = OpeningBook()
    book.set_path(path)
    move = book.probe(Board(), best=True)
    assert move is not None
    assert move_to_uci(move) in {"e2e4", "d2d4", "c2c4", "g1f3"}


def test_opening_book_picks_the_highest_weight(tmp_path: Path) -> None:
    path = tmp_path / "book.bin"
    _write_book(path, chess.STARTING_FEN, [("e2e4", 50), ("d2d4", 10)])
    book = OpeningBook()
    book.set_path(path)
    move = book.probe(Board(), best=True)
    assert move is not None
    assert move_to_uci(move) == "e2e4"


def test_own_book_plays_on_the_clock_not_at_fixed_depth(tmp_path: Path) -> None:
    path = tmp_path / "book.bin"
    _write_book(path, chess.STARTING_FEN, [("d2d4", 99)])
    clock = run_commands(
        "isready",
        f"setoption name BookFile value {path}",
        "setoption name OwnBook value true",
        "position startpos",
        "go wtime 8000 btime 8000",
    )
    assert "info string book" in clock
    assert clock[-1] == "bestmove d2d4"
    depth = run_commands(
        "isready",
        f"setoption name BookFile value {path}",
        "setoption name OwnBook value true",
        "position startpos",
        "go depth 3",
    )
    assert "info string book" not in depth
    assert any(line.startswith("info depth 3 ") for line in depth)
    assert depth[-1].startswith("bestmove ")


def test_own_book_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "book.bin"
    _write_book(path, chess.STARTING_FEN, [("a2a4", 99)])
    lines = run_commands(
        "isready",
        f"setoption name BookFile value {path}",
        "setoption name OwnBook value false",
        "position startpos",
        "go wtime 1500 btime 1500",
    )
    assert "info string book" not in lines
    assert any(line.startswith("info depth ") for line in lines)


def test_uci_advertises_book_and_syzygy_options() -> None:
    lines = run_commands("uci")
    joined = "\n".join(lines)
    assert "option name OwnBook type check default true" in joined
    assert "option name BookFile type string" in joined
    assert "option name SyzygyPath type string" in joined
    assert "option name SyzygyProbeLimit type spin default 5" in joined


def test_jitted_packages_do_not_import_python_chess() -> None:
    root = Path(chessbot.__file__).parent
    pattern = re.compile(r"^(?:from|import) chess(?:\.| |$)", re.M)
    for pkg in ("core", "search"):
        for path in sorted((root / pkg).glob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert pattern.search(text) is None, path


def test_book_builder_round_trips_a_pgn(tmp_path: Path) -> None:
    pgn = tmp_path / "g.pgn"
    pgn.write_text(
        '[WhiteElo "2600"]\n[BlackElo "2550"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0\n"
    )
    from chessbot.tools.book import collect_entries

    counts = collect_entries([pgn], max_ply=8, min_rating=2500)
    dest = tmp_path / "out.bin"
    n = write_book(counts, dest, min_games=1)
    assert n >= 1
    book = OpeningBook()
    book.set_path(dest)
    assert move_to_uci(book.probe(Board(), best=True) or 0) == "e2e4"


@pytest.mark.skipif(not (SYZYGY_3 / "KQvK.rtbw").is_file(), reason=NEEDS_TB)
def test_syzygy_converts_mate_in_one() -> None:
    tb = Tablebases()
    assert tb.set_path(str(SYZYGY_3)) > 0
    hit = tb.probe_dtz_move(Board(KQK_MATE))
    assert hit is not None
    move, score = hit
    assert score > 0
    assert _push_engine_move(KQK_MATE, move).is_checkmate()


@pytest.mark.skipif(not (SYZYGY_3 / "KRvK.rtbw").is_file(), reason=NEEDS_TB)
def test_syzygy_krk_stays_winning() -> None:
    tb = Tablebases()
    tb.set_path(str(SYZYGY_3))
    hit = tb.probe_dtz_move(Board(KRK_WIN))
    assert hit is not None
    move, score = hit
    assert score > 0
    with chess.syzygy.open_tablebase(str(SYZYGY_3)) as tables:
        assert tables.probe_wdl(_push_engine_move(KRK_WIN, move)) == -2


@pytest.mark.skipif(not (SYZYGY_3 / "KPvK.rtbw").is_file(), reason=NEEDS_TB)
def test_syzygy_kpk_is_a_win() -> None:
    assert piece_count(KPK_WIN) == 3
    tb = Tablebases()
    tb.set_path(str(SYZYGY_3))
    hit = tb.probe_dtz_move(Board(KPK_WIN))
    assert hit is not None
    move, score = hit
    assert score > 0
    with chess.syzygy.open_tablebase(str(SYZYGY_3)) as tables:
        assert tables.probe_wdl(_push_engine_move(KPK_WIN, move)) == -2


@pytest.mark.skipif(not (SYZYGY_3 / "KQvK.rtbw").is_file(), reason=NEEDS_TB)
def test_uci_dtz_at_the_root() -> None:
    lines = run_commands(
        "isready",
        f"setoption name SyzygyPath value {SYZYGY_3}",
        "setoption name OwnBook value false",
        f"position fen {KQK_MATE}",
        "go depth 4",
    )
    assert "info string syzygy" in lines
    best = lines[-1].split()[1]
    board = chess.Board(KQK_MATE)
    board.push_uci(best)
    assert board.is_checkmate()
