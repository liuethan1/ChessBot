"""Test-suite runner: solve count for EPD suites such as WAC (bm/am) and STS (bm plus points)."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from chessbot.tools.common import parse_options, split_command
from chessbot.tools.epd import SuiteEntry, read_test_suite


@dataclass
class SuiteOutcome:
    id: str
    move: str | None
    solved: bool
    points: int
    max_points: int


def score(entry: SuiteEntry, move: chess.Move | None) -> SuiteOutcome:
    solved = move is not None and move not in entry.avoid
    if entry.best:
        solved = solved and move in entry.best
    points = entry.points.get(move, 0) if move is not None else 0
    max_points = max(entry.points.values(), default=0)
    return SuiteOutcome(entry.id, move.uci() if move else None, solved, points, max_points)


def _worker(
    command: str,
    entries: list[SuiteEntry],
    limit: chess.engine.Limit,
    options: dict[str, str],
) -> list[SuiteOutcome]:
    engine = chess.engine.SimpleEngine.popen_uci(split_command(command))
    try:
        engine.configure(options)
        outcomes = []
        for entry in entries:
            result = engine.play(entry.board, limit, game=object())
            outcomes.append(score(entry, result.move))
        return outcomes
    finally:
        engine.quit()


def run_suite(
    command: str,
    entries: list[SuiteEntry],
    limit: chess.engine.Limit,
    options: dict[str, str] | None = None,
    concurrency: int = 1,
) -> list[SuiteOutcome]:
    chunks = [entries[i::concurrency] for i in range(concurrency)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, command, c, limit, options or {}) for c in chunks if c]
        by_id = {o.id: o for f in futures for o in f.result()}
    return [by_id[e.id] for e in entries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite", type=Path, help="EPD file")
    parser.add_argument("--engine", required=True, help="engine command line")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--movetime", type=float, default=1.0, help="seconds per position")
    limits.add_argument("--depth", type=int)
    limits.add_argument("--nodes", type=int)
    parser.add_argument("--option", action="append", metavar="NAME=VALUE", help="UCI option")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only run the first N positions")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--json", type=Path, help="also write per-position results as JSON")
    args = parser.parse_args(argv)

    if args.depth is not None:
        limit = chess.engine.Limit(depth=args.depth)
    elif args.nodes is not None:
        limit = chess.engine.Limit(nodes=args.nodes)
    else:
        limit = chess.engine.Limit(time=args.movetime)

    entries = read_test_suite(args.suite)[: args.limit]
    outcomes = run_suite(args.engine, entries, limit, parse_options(args.option), args.concurrency)
    solved = sum(o.solved for o in outcomes)
    print(f"{args.suite.name}: solved {solved}/{len(outcomes)}")
    max_points = sum(o.max_points for o in outcomes)
    if max_points:
        points = sum(o.points for o in outcomes)
        print(f"points {points}/{max_points} ({100 * points / max_points:.1f}%)")
    if args.show_failures:
        for o in outcomes:
            if not o.solved:
                print(f"  failed {o.id}: played {o.move}")
    if args.json:
        args.json.write_text(json.dumps([o.__dict__ for o in outcomes], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
