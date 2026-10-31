"""SPRT calculator (GSPRT approximation, logistic Elo) and Elo estimate from match results.

Works from trinomial (W/D/L) or pentanomial (game-pair) counts, or from a PGN of finished games.
fastchess and cutechess-cli report the same numbers live; this is for re-checking results,
combining runs, or evaluating a PGN produced elsewhere.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import chess.pgn


def expected_score(elo: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Lower and upper LLR bounds: accept H0 at or below the first, H1 at or above the second."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


@dataclass(frozen=True)
class Stats:
    n: int
    """Number of samples (games for trinomial, game pairs for pentanomial)."""
    mean: float
    """Mean score per game, in [0, 1]."""
    variance: float
    """Variance of one sample's per-game score."""
    values: tuple[float, ...] = ()
    counts: tuple[int, ...] = ()


def _stats(values: list[float], counts: list[int]) -> Stats:
    n = sum(counts)
    if n == 0:
        return Stats(0, 0.5, 0.0)
    mean = sum(v * c for v, c in zip(values, counts, strict=True)) / n
    variance = sum(c * (v - mean) ** 2 for v, c in zip(values, counts, strict=True)) / n
    return Stats(n, mean, variance, tuple(values), tuple(counts))


def trinomial(wins: int, draws: int, losses: int) -> Stats:
    return _stats([1.0, 0.5, 0.0], [wins, draws, losses])


def pentanomial(counts: list[int]) -> Stats:
    """``counts`` are game pairs scoring 0, 0.5, 1, 1.5, and 2 points, in that order."""
    if len(counts) != 5:
        raise ValueError("pentanomial needs exactly 5 counts")
    return _stats([0.0, 0.25, 0.5, 0.75, 1.0], counts)


def _mle_with_mean(values: tuple[float, ...], probs: list[float], target: float) -> list[float]:
    """Closest distribution (max likelihood) to ``probs`` on ``values`` with the given mean."""
    shifted = [v - target for v in values]
    # The MLE is probs[i] / (1 + x * shifted[i]); x solves sum(p * a / (1 + x * a)) = 0, which is
    # decreasing in x on the interval where every denominator stays positive.
    lo = -1 / max(shifted) if max(shifted) > 0 else -1e9
    hi = -1 / min(shifted) if min(shifted) < 0 else 1e9
    for _ in range(200):
        x = (lo + hi) / 2
        total = sum(p * a / (1 + x * a) for p, a in zip(probs, shifted, strict=True))
        if total > 0:
            lo = x
        else:
            hi = x
    x = (lo + hi) / 2
    return [p / (1 + x * a) for p, a in zip(probs, shifted, strict=True)]


def llr(stats: Stats, elo0: float, elo1: float) -> float:
    """Generalized SPRT log-likelihood ratio, computed as fishtest and fastchess do.

    fastchess solves for theta only to a 1e-3 tolerance, so on very small or lopsided samples its
    printed LLR can differ from this one; at normal SPRT sample sizes they agree.
    """
    if stats.n == 0 or stats.variance <= 0:
        return 0.0
    counts = [c if c > 0 else 1e-3 for c in stats.counts]
    total = sum(counts)
    probs = [c / total for c in counts]
    s0, s1 = expected_score(elo0), expected_score(elo1)
    p0 = _mle_with_mean(stats.values, probs, s0)
    p1 = _mle_with_mean(stats.values, probs, s1)
    per_sample = sum(p * (math.log(b) - math.log(a)) for p, a, b in zip(probs, p0, p1, strict=True))
    return total * per_sample


def elo_estimate(stats: Stats) -> tuple[float, float]:
    """Elo difference and the half-width of its 95% confidence interval."""
    elo = elo_from_score(stats.mean)
    if stats.n == 0:
        return elo, math.inf
    margin = 1.959964 * math.sqrt(stats.variance / stats.n)
    upper = elo_from_score(stats.mean + margin)
    lower = elo_from_score(stats.mean - margin)
    return elo, (upper - lower) / 2


@dataclass(frozen=True)
class SprtResult:
    llr: float
    lower: float
    upper: float
    elo: float
    elo_error: float

    @property
    def verdict(self) -> str:
        if self.llr >= self.upper:
            return "H1 accepted (passed)"
        if self.llr <= self.lower:
            return "H0 accepted (failed)"
        return "continue"


def sprt(stats: Stats, elo0: float, elo1: float, alpha: float, beta: float) -> SprtResult:
    lower, upper = bounds(alpha, beta)
    elo, error = elo_estimate(stats)
    return SprtResult(llr(stats, elo0, elo1), lower, upper, elo, error)


def wdl_from_pgn(path: Path, engine: str) -> tuple[int, int, int]:
    wins = draws = losses = 0
    with path.open() as handle:
        while (headers := chess.pgn.read_headers(handle)) is not None:
            result = headers.get("Result", "*")
            if result == "*" or engine not in (headers.get("White"), headers.get("Black")):
                continue
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == (headers.get("White") == engine):
                wins += 1
            else:
                losses += 1
    return wins, draws, losses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wdl", nargs=3, type=int, metavar=("W", "D", "L"))
    source.add_argument("--penta", nargs=5, type=int, metavar=("LL", "LD", "DD/WL", "WD", "WW"))
    source.add_argument("--pgn", type=Path, help="PGN of finished games (trinomial)")
    parser.add_argument("--engine", help="engine name to score in --pgn")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args(argv)

    if args.pgn:
        if not args.engine:
            parser.error("--pgn needs --engine")
        wins, draws, losses = wdl_from_pgn(args.pgn, args.engine)
        print(f"W/D/L for {args.engine}: {wins}/{draws}/{losses}")
        stats = trinomial(wins, draws, losses)
    elif args.wdl:
        stats = trinomial(*args.wdl)
    else:
        stats = pentanomial(args.penta)

    result = sprt(stats, args.elo0, args.elo1, args.alpha, args.beta)
    print(f"Elo      {result.elo:+.1f} +/- {result.elo_error:.1f} (95%)")
    print(
        f"LLR      {result.llr:.2f} ({result.lower:.2f}, {result.upper:.2f}) "
        f"[{args.elo0:g}, {args.elo1:g}]"
    )
    print(f"verdict  {result.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
