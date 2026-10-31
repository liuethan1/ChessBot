"""Benchmark: search a fixed set of positions to a fixed depth over UCI.

The total node count is a functional fingerprint (it changes only when search behaviour changes,
so record it in commit messages), and nodes per second measures speed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine

from chessbot.tools.common import data_path, parse_options, split_command


@dataclass
class BenchPosition:
    fen: str
    nodes: int
    seconds: float
    best_move: str | None


@dataclass
class BenchResult:
    engine: str
    depth: int
    positions: list[BenchPosition]

    @property
    def nodes(self) -> int:
        return sum(p.nodes for p in self.positions)

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.positions)

    @property
    def nps(self) -> float:
        return self.nodes / self.seconds if self.seconds > 0 else 0.0


def read_fens(path: Path) -> list[str]:
    fens = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            fens.append(chess.Board(line).fen())
    return fens


def run_bench(
    command: str,
    depth: int,
    fens: list[str],
    options: dict[str, str] | None = None,
) -> BenchResult:
    engine = chess.engine.SimpleEngine.popen_uci(split_command(command))
    try:
        config: dict[str, str | int] = dict(options or {})
        if "Threads" in engine.options and "Threads" not in config:
            config["Threads"] = 1
        engine.configure(config)
        positions = []
        for fen in fens:
            start = time.perf_counter()
            info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth), game=object())
            elapsed = time.perf_counter() - start
            pv = info.get("pv")
            positions.append(
                BenchPosition(fen, int(info.get("nodes", 0)), elapsed, pv[0].uci() if pv else None)
            )
        return BenchResult(engine.id.get("name", command), depth, positions)
    finally:
        engine.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, help="engine command line")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--positions", type=Path, default=data_path("bench.epd"))
    parser.add_argument("--option", action="append", metavar="NAME=VALUE", help="UCI option")
    parser.add_argument("--json", type=Path, help="also write results as JSON")
    parser.add_argument("--verbose", action="store_true", help="print each position")
    args = parser.parse_args(argv)

    result = run_bench(
        args.engine, args.depth, read_fens(args.positions), parse_options(args.option)
    )
    if args.verbose:
        for p in result.positions:
            print(f"{p.nodes:>12} nodes {p.seconds:7.3f}s  {p.best_move or '-':6} {p.fen}")
    print(f"engine   {result.engine}")
    print(f"depth    {result.depth}")
    print(f"nodes    {result.nodes}")
    print(f"time     {result.seconds:.3f}s")
    print(f"nps      {result.nps:.0f}")
    if args.json:
        payload = asdict(result) | {"nodes": result.nodes, "nps": result.nps}
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
