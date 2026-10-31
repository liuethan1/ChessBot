"""Average centipawn loss (ACPL) per side for PGN games, using Stockfish.

Each position is evaluated once to a fixed depth, from the side to move's point of view. The loss
of a move is ``eval_before + eval_after`` (the eval after the move is from the opponent's side),
floored at 0. Evals are clamped to +/- ``cap`` and each move's loss to ``max_loss`` (both 1000 cp
by default), so a single blunder costs at most one whole decisive advantage and missed mates do not
dominate the average. With these defaults the legacy bot's self-play game
(``legacy/legacy-selfplay-game.pgn``) scores about 360 ACPL at depth 14.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chessbot.tools.common import find_binary

BLUNDER = 300
MISTAKE = 100
INACCURACY = 50

Evaluator = Callable[[chess.Board], int]


@dataclass
class SideReport:
    moves: int = 0
    total_loss: int = 0
    blunders: int = 0
    mistakes: int = 0
    inaccuracies: int = 0
    losses: list[int] = field(default_factory=list)

    @property
    def acpl(self) -> float:
        return self.total_loss / self.moves if self.moves else 0.0

    def add(self, loss: int) -> None:
        self.moves += 1
        self.total_loss += loss
        self.losses.append(loss)
        if loss >= BLUNDER:
            self.blunders += 1
        elif loss >= MISTAKE:
            self.mistakes += 1
        elif loss >= INACCURACY:
            self.inaccuracies += 1


@dataclass
class GameReport:
    white: SideReport
    black: SideReport
    headers: dict[str, str]


def terminal_eval(board: chess.Board, cap: int) -> int | None:
    if board.is_checkmate():
        return -cap
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return None


def analyse_game(
    game: chess.pgn.Game, evaluate: Evaluator, cap: int = 1000, max_loss: int = 1000
) -> GameReport:
    board = game.board()
    evals = [_clamped(board, evaluate, cap)]
    movers = []
    for move in game.mainline_moves():
        movers.append(board.turn)
        board.push(move)
        evals.append(_clamped(board, evaluate, cap))

    white, black = SideReport(), SideReport()
    for i, mover in enumerate(movers):
        loss = min(max_loss, max(0, evals[i] + evals[i + 1]))
        (white if mover == chess.WHITE else black).add(loss)
    return GameReport(white, black, dict(game.headers))


def _clamped(board: chess.Board, evaluate: Evaluator, cap: int) -> int:
    terminal = terminal_eval(board, cap)
    score = terminal if terminal is not None else evaluate(board)
    return max(-cap, min(cap, score))


class StockfishEvaluator:
    def __init__(self, path: str, depth: int, hash_mb: int = 64) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": hash_mb})
        self.limit = chess.engine.Limit(depth=depth)

    def __call__(self, board: chess.Board) -> int:
        info = self.engine.analyse(board, self.limit)
        return info["score"].relative.score(mate_score=100_000)

    def close(self) -> None:
        self.engine.quit()


def analyse_pgn(
    path: Path, evaluator: Evaluator, cap: int = 1000, max_loss: int = 1000
) -> list[GameReport]:
    reports = []
    with path.open() as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            reports.append(analyse_game(game, evaluator, cap, max_loss))
    return reports


def combined_acpl(report: GameReport) -> float:
    moves = report.white.moves + report.black.moves
    return (report.white.total_loss + report.black.total_loss) / moves if moves else 0.0


def format_report(report: GameReport) -> str:
    title = f"{report.headers.get('White', '?')} vs {report.headers.get('Black', '?')}"
    lines = [title, "side   moves   ACPL  blunders  mistakes  inaccuracies"]
    for name, side in (("white", report.white), ("black", report.black)):
        lines.append(
            f"{name:5} {side.moves:6} {side.acpl:6.0f} {side.blunders:9} "
            f"{side.mistakes:9} {side.inaccuracies:13}"
        )
    lines.append(f"both  {report.white.moves + report.black.moves:6} {combined_acpl(report):6.0f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--cap", type=int, default=1000, help="clamp evals to +/- this many cp")
    parser.add_argument("--max-loss", type=int, default=1000, help="cap each move's loss")
    parser.add_argument("--stockfish", help="Stockfish binary (default: STOCKFISH_PATH or PATH)")
    parser.add_argument("--json", type=Path, help="also write results (with per-move losses)")
    args = parser.parse_args(argv)

    path = args.stockfish or find_binary("stockfish", "STOCKFISH_PATH")
    if path is None:
        parser.error("Stockfish not found; install it or pass --stockfish / set STOCKFISH_PATH")
    evaluator = StockfishEvaluator(path, args.depth)
    try:
        reports = analyse_pgn(args.pgn, evaluator, args.cap, args.max_loss)
    finally:
        evaluator.close()

    print(
        f"Stockfish depth {args.depth}, evals clamped to +/-{args.cap} cp, "
        f"loss per move capped at {args.max_loss} cp\n"
    )
    print("\n\n".join(format_report(r) for r in reports))
    if args.json:
        payload = [
            {"headers": r.headers}
            | {
                side: asdict(getattr(r, side)) | {"acpl": getattr(r, side).acpl}
                for side in ("white", "black")
            }
            for r in reports
        ]
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
