from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from chessbot.tools.common import REPO_ROOT, find_binary

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def stockfish() -> str:
    path = find_binary("stockfish", "STOCKFISH_PATH")
    if path is None:
        pytest.skip("Stockfish not found")
    return path


@pytest.fixture(scope="session")
def fastchess() -> str:
    path = find_binary("fastchess", "FASTCHESS_PATH")
    if path is None:
        pytest.skip("fastchess not found")
    return path


@pytest.fixture(scope="session")
def greedy_engine() -> str:
    """Command line for the deterministic test engine in fixtures/greedy_uci.py."""
    return shlex.join([sys.executable, str(FIXTURES / "greedy_uci.py")])


@pytest.fixture(scope="session")
def legacy_game() -> Path:
    return REPO_ROOT / "legacy" / "legacy-selfplay-game.pgn"
