"""Engine-vs-engine matches (SPRT or fixed length) through fastchess or cutechess-cli.

Typical uses:

* SPRT a change:  ``--new "python -m chessbot" --base "<previous build>" --sprt 0 5``
* Gauntlet vs a rated reference:
  ``--base stockfish --base-option UCI_LimitStrength=true --base-option UCI_Elo=1500 --games 200``
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from chessbot.tools.common import EXTERNAL_DIR, find_binary, parse_options, split_command

DEFAULT_BOOK = EXTERNAL_DIR / "books" / "8moves_v3.pgn"
RUNNERS = ("fastchess", "cutechess-cli")


@dataclass
class EngineSpec:
    command: str
    name: str
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class MatchConfig:
    new: EngineSpec
    base: EngineSpec
    tc: str = "8+0.08"
    nodes: int | None = None
    book: Path | None = None
    games: int = 1000
    concurrency: int = 1
    sprt: tuple[float, float] | None = None
    alpha: float = 0.05
    beta: float = 0.05
    model: str = "logistic"
    pgnout: Path | None = None
    seed: int | None = None
    adjudicate: bool = True
    extra: list[str] = field(default_factory=list)


def _engine_args(runner: str, engine: EngineSpec) -> list[str]:
    cmd, *rest = split_command(engine.command)
    args = ["-engine", f"cmd={cmd}", f"name={engine.name}"]
    if rest:
        if runner == "fastchess":
            args.append(f"args={shlex.join(rest)}")
        else:
            args += [f"arg={a}" for a in rest]
    args += [f"option.{k}={v}" for k, v in engine.options.items()]
    return args


def build_command(runner: str, binary: str, config: MatchConfig) -> list[str]:
    if runner not in RUNNERS:
        raise ValueError(f"runner must be one of {RUNNERS}")
    cmd = [binary]
    cmd += _engine_args(runner, config.new)
    cmd += _engine_args(runner, config.base)

    each = ["proto=uci"]
    if config.nodes is not None:
        each += ["tc=inf", f"nodes={config.nodes}"]
    else:
        each.append(f"tc={config.tc}")
    cmd += ["-each", *each]

    if config.book is not None:
        book_format = "pgn" if config.book.suffix == ".pgn" else "epd"
        cmd += ["-openings", f"file={config.book}", f"format={book_format}", "order=random"]
    cmd += ["-repeat", "-rounds", str(max(1, config.games // 2)), "-games", "2"]
    cmd += ["-concurrency", str(config.concurrency)]

    if config.sprt is not None:
        elo0, elo1 = config.sprt
        cmd += ["-sprt", f"elo0={elo0:g}", f"elo1={elo1:g}"]
        cmd += [f"alpha={config.alpha:g}", f"beta={config.beta:g}"]
        if runner == "fastchess":
            cmd.append(f"model={config.model}")

    if config.adjudicate:
        cmd += ["-draw", "movenumber=40", "movecount=8", "score=10"]
        cmd += ["-resign", "movecount=3", "score=600"]
    if config.pgnout is not None:
        if runner == "fastchess":
            cmd += ["-pgnout", f"file={config.pgnout}"]
        else:
            cmd += ["-pgnout", str(config.pgnout)]
    if config.seed is not None:
        cmd += ["-srand", str(config.seed)]
    if runner == "fastchess":
        cmd += ["-report", "penta=true"]
    cmd += ["-recover", *config.extra]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", required=True, help="command line of the engine under test")
    parser.add_argument("--base", required=True, help="command line of the baseline engine")
    parser.add_argument("--new-name", default="new")
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--new-option", action="append", metavar="NAME=VALUE")
    parser.add_argument("--base-option", action="append", metavar="NAME=VALUE")
    parser.add_argument("--tc", default="8+0.08", help="time control, e.g. 8+0.08 or 40+0.4")
    parser.add_argument("--nodes", type=int, help="fixed nodes per move instead of a clock")
    parser.add_argument("--book", type=Path, help=f"opening book (default {DEFAULT_BOOK})")
    parser.add_argument("--no-book", action="store_true")
    parser.add_argument("--games", type=int, default=1000, help="maximum number of games")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument(
        "--model", default="logistic", choices=["logistic", "normalized", "bayesian"]
    )
    parser.add_argument("--pgnout", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-adjudication", action="store_true")
    parser.add_argument("--runner", choices=RUNNERS, default="fastchess")
    parser.add_argument("--runner-path", help="path to the fastchess/cutechess-cli binary")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args, extra = parser.parse_known_args(argv)

    book = None
    if not args.no_book:
        book = args.book or DEFAULT_BOOK
        if not book.exists() and not args.dry_run:
            parser.error(f"opening book {book} not found (run `chessbot-fetch books` or --no-book)")

    config = MatchConfig(
        new=EngineSpec(args.new, args.new_name, parse_options(args.new_option)),
        base=EngineSpec(args.base, args.base_name, parse_options(args.base_option)),
        tc=args.tc,
        nodes=args.nodes,
        book=book,
        games=args.games,
        concurrency=args.concurrency,
        sprt=tuple(args.sprt) if args.sprt else None,
        alpha=args.alpha,
        beta=args.beta,
        model=args.model,
        pgnout=args.pgnout,
        seed=args.seed,
        adjudicate=not args.no_adjudication,
        extra=extra,
    )
    env_var = "FASTCHESS_PATH" if args.runner == "fastchess" else "CUTECHESS_PATH"
    binary = args.runner_path or find_binary(args.runner, env_var)
    if binary is None and not args.dry_run:
        parser.error(f"{args.runner} not found (run `chessbot-fetch fastchess` or set {env_var})")
    command = build_command(args.runner, binary or args.runner, config)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
