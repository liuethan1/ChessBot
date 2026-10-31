from __future__ import annotations

import chess
import pytest

from chessbot.tools.common import data_path
from chessbot.tools.epd import read_perft_suite
from chessbot.tools.perft import (
    FunctionBackend,
    ReferenceBackend,
    find_mismatch,
    main,
    make_backend,
    run_suite,
)

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def test_suites_parse() -> None:
    assert len(read_perft_suite(data_path("perft", "cpw.epd"))) == 6
    standard = read_perft_suite(data_path("perft", "standard.epd"))
    assert len(standard) >= 126
    assert standard[0].counts[6] == 119_060_324


def test_reference_matches_both_suites_at_small_node_counts() -> None:
    for suite in ("cpw.epd", "standard.epd"):
        entries = read_perft_suite(data_path("perft", suite))
        results = run_suite(ReferenceBackend(), entries, max_depth=6, max_nodes=20_000)
        assert results
        assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_divide_sums_to_perft() -> None:
    backend = ReferenceBackend()
    divide = backend.divide(KIWIPETE, 2)
    assert len(divide) == 48
    assert sum(divide.values()) == 2039


def buggy_perft(fen: str, depth: int) -> int:
    """Reference perft that forgets every en passant capture."""

    def count(board: chess.Board, d: int) -> int:
        if d == 0:
            return 1
        total = 0
        for move in board.legal_moves:
            if board.is_en_passant(move):
                continue
            board.push(move)
            total += count(board, d - 1)
            board.pop()
        return total

    return count(chess.Board(fen), depth)


def test_find_mismatch_locates_missing_en_passant() -> None:
    backend = FunctionBackend(f"{__name__}:buggy_perft")
    assert backend.perft(KIWIPETE, 3) != ReferenceBackend().perft(KIWIPETE, 3)
    mismatch = find_mismatch(backend, KIWIPETE, 3)
    assert mismatch is not None
    assert len(mismatch.path) >= 1
    board = chess.Board(mismatch.fen)
    assert any(board.is_en_passant(m) for m in board.legal_moves)
    assert mismatch.total[0] < mismatch.total[1]
    assert "move generation is wrong here" in mismatch.describe()


class DropsCastlingUci:
    """A backend whose divide output (like a UCI engine's) omits castling moves."""

    name = "drops-castling"

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        board = chess.Board(fen)
        result = ReferenceBackend().divide(fen, depth)
        return {m: n for m, n in result.items() if not board.is_castling(chess.Move.from_uci(m))}

    def perft(self, fen: str, depth: int) -> int:
        return sum(self.divide(fen, depth).values())

    def close(self) -> None:
        pass


def test_find_mismatch_reports_missing_moves() -> None:
    mismatch = find_mismatch(DropsCastlingUci(), KIWIPETE, 2)
    assert mismatch is not None
    assert mismatch.path == []
    assert sorted(mismatch.missing) == ["e1c1", "e1g1"]
    assert "missing moves: e1c1 e1g1" in mismatch.describe()


def test_find_mismatch_returns_none_when_correct() -> None:
    assert find_mismatch(ReferenceBackend(), KIWIPETE, 2) is None


def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--max-nodes", "10000"]) == 0
    assert "6/6 passed" in capsys.readouterr().out
    assert main(["--backend", f"py:{__name__}:buggy_perft", "--max-nodes", "100000"]) == 1
    assert "First difference" in capsys.readouterr().out


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("nope")


@pytest.mark.stockfish
def test_uci_backend_against_stockfish(stockfish: str) -> None:
    backend = make_backend(f"uci:{stockfish}")
    try:
        assert backend.perft(KIWIPETE, 3) == 97_862
        assert backend.divide(KIWIPETE, 2) == ReferenceBackend().divide(KIWIPETE, 2)
    finally:
        backend.close()
