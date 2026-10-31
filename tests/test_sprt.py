from __future__ import annotations

import math
from pathlib import Path

import pytest

from chessbot.tools import sprt


def test_bounds() -> None:
    lower, upper = sprt.bounds(0.05, 0.05)
    assert lower == pytest.approx(-2.944, abs=1e-3)
    assert upper == pytest.approx(2.944, abs=1e-3)


def test_elo_score_round_trip() -> None:
    for elo in (-300.0, -5.0, 0.0, 5.0, 300.0):
        assert sprt.elo_from_score(sprt.expected_score(elo)) == pytest.approx(elo)


def test_even_match_has_zero_elo() -> None:
    result = sprt.sprt(sprt.trinomial(100, 200, 100), 0, 5, 0.05, 0.05)
    assert result.elo == pytest.approx(0.0)
    assert result.llr < 0
    assert result.verdict == "continue"


def test_matches_fastchess_reference_values() -> None:
    # Elo +/- error printed by fastchess for this pentanomial result (logistic model, 60 games).
    stats = sprt.pentanomial([4, 1, 11, 0, 14])
    elo, error = sprt.elo_estimate(stats)
    assert elo == pytest.approx(113.94, abs=0.01)
    assert error == pytest.approx(102.22, abs=0.01)


def test_llr_matches_variance_approximation_on_large_samples() -> None:
    stats = sprt.pentanomial([400, 1000, 2200, 1100, 450])
    s0, s1 = sprt.expected_score(0), sprt.expected_score(5)
    approx = stats.n * (s1 - s0) * (2 * stats.mean - s0 - s1) / (2 * stats.variance)
    assert sprt.llr(stats, 0, 5) == pytest.approx(approx, rel=0.01)
    assert sprt.sprt(stats, 0, 5, 0.05, 0.05).verdict == "H1 accepted (passed)"


def test_clear_loss_is_rejected() -> None:
    result = sprt.sprt(sprt.trinomial(100, 300, 400), 0, 5, 0.05, 0.05)
    assert result.verdict == "H0 accepted (failed)"


def test_degenerate_inputs() -> None:
    assert sprt.llr(sprt.trinomial(0, 0, 0), 0, 5) == 0.0
    assert sprt.llr(sprt.trinomial(0, 10, 0), 0, 5) == 0.0
    assert math.isinf(sprt.elo_estimate(sprt.trinomial(0, 0, 0))[1])
    with pytest.raises(ValueError, match="5 counts"):
        sprt.pentanomial([1, 2, 3])


def test_wdl_from_pgn(tmp_path: Path) -> None:
    pgn = tmp_path / "games.pgn"
    games = [
        ("new", "base", "1-0"),
        ("base", "new", "1-0"),
        ("new", "base", "1/2-1/2"),
        ("base", "new", "0-1"),
        ("other", "base", "1-0"),
        ("new", "base", "*"),
    ]
    pgn.write_text(
        "".join(f'[White "{w}"]\n[Black "{b}"]\n[Result "{r}"]\n\n{r}\n\n' for w, b, r in games)
    )
    assert sprt.wdl_from_pgn(pgn, "new") == (2, 1, 1)


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert sprt.main(["--wdl", "450", "400", "150", "--elo0", "0", "--elo1", "5"]) == 0
    assert "H1 accepted" in capsys.readouterr().out
