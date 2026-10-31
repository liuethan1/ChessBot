from __future__ import annotations

from pathlib import Path

import pytest

from chessbot.tools import match


def _config(**overrides: object) -> match.MatchConfig:
    config = match.MatchConfig(
        new=match.EngineSpec("python -m chessbot", "new", {"Hash": "16"}),
        base=match.EngineSpec("/usr/games/stockfish", "base", {"UCI_Elo": "1500"}),
        book=Path("book.pgn"),
        games=100,
        concurrency=4,
        sprt=(0.0, 5.0),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _option_after(cmd: list[str], flag: str) -> list[str]:
    i = cmd.index(flag)
    end = next((j for j in range(i + 1, len(cmd)) if cmd[j].startswith("-")), len(cmd))
    return cmd[i + 1 : end]


def test_fastchess_command() -> None:
    cmd = match.build_command("fastchess", "fastchess", _config())
    assert cmd[:6] == [
        "fastchess",
        "-engine",
        "cmd=python",
        "name=new",
        "args=-m chessbot",
        "option.Hash=16",
    ]
    assert "option.UCI_Elo=1500" in cmd
    assert _option_after(cmd, "-each") == ["proto=uci", "tc=8+0.08"]
    assert _option_after(cmd, "-openings") == ["file=book.pgn", "format=pgn", "order=random"]
    assert _option_after(cmd, "-rounds") == ["50"]
    assert _option_after(cmd, "-sprt") == [
        "elo0=0",
        "elo1=5",
        "alpha=0.05",
        "beta=0.05",
        "model=logistic",
    ]
    assert "-recover" in cmd


def test_cutechess_command_splits_engine_args() -> None:
    cmd = match.build_command("cutechess-cli", "cutechess-cli", _config(pgnout=Path("out.pgn")))
    assert cmd[1:6] == ["-engine", "cmd=python", "name=new", "arg=-m", "arg=chessbot"]
    assert "model=logistic" not in cmd
    assert _option_after(cmd, "-pgnout") == ["out.pgn"]


def test_fixed_nodes_and_epd_book() -> None:
    cmd = match.build_command(
        "fastchess", "fastchess", _config(nodes=5000, book=Path("uho.epd"), sprt=None)
    )
    assert _option_after(cmd, "-each") == ["proto=uci", "tc=inf", "nodes=5000"]
    assert "format=epd" in cmd
    assert "-sprt" not in cmd


def test_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert match.main(["--new", "a", "--base", "b", "--no-book", "--dry-run"]) == 0
    assert "-engine cmd=a name=new" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.stockfish
@pytest.mark.fastchess
def test_real_match(fastchess: str, stockfish: str, greedy_engine: str, tmp_path: Path) -> None:
    pgn = tmp_path / "games.pgn"
    code = match.main(
        [
            "--new",
            stockfish,
            "--new-name",
            "sf",
            "--base",
            greedy_engine,
            "--base-name",
            "greedy",
            "--nodes",
            "2000",
            "--games",
            "4",
            "--no-book",
            "--seed",
            "1",
            "--sprt",
            "0",
            "100",
            "--pgnout",
            str(pgn),
            "--runner-path",
            fastchess,
        ]
    )
    assert code == 0
    from chessbot.tools.sprt import wdl_from_pgn

    wins, draws, losses = wdl_from_pgn(pgn, "sf")
    assert wins + draws + losses == 4
    assert wins >= 3
