"""UCI protocol loop (``python -m chessbot``).

The jitted code is loaded on a background thread as soon as ``uci`` arrives, so ``uciok`` is
immediate; ``isready``, ``position``, and ``go`` wait for it, which keeps numba's compile or cache
load off the clock. Searches run on a worker thread so ``stop``, ``isready``, and ``quit`` are
answered while thinking.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from chessbot import __version__
from chessbot.core.board import Board
from chessbot.core.position import START_FEN, move_to_uci
from chessbot.search import Limits, Searcher, SearchInfo, SearchResult, warmup

ENGINE_NAME = f"ChessBot {__version__}"
AUTHOR = "Ethan Liu"
MIN_ELO, MAX_ELO = 1320, 3190

GO_INT_FIELDS = ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth", "nodes")


def format_info(info: SearchInfo) -> str:
    mate = info.mate_in
    score = f"mate {mate}" if mate is not None else f"cp {info.score}"
    ms = int(info.seconds * 1000)
    nps = int(info.nodes / info.seconds) if info.seconds > 0 else 0
    pv = " ".join(move_to_uci(m) for m in info.pv)
    return (
        f"info depth {info.depth} seldepth {info.seldepth} score {score} nodes {info.nodes} "
        f"nps {nps} time {ms} pv {pv}"
    )


def parse_go(tokens: list[str]) -> Limits:
    limits = Limits()
    i = 0
    while i < len(tokens):
        name = tokens[i]
        if name in GO_INT_FIELDS and i + 1 < len(tokens):
            setattr(limits, name, int(tokens[i + 1]))
            i += 2
            continue
        if name == "infinite":
            limits.infinite = True
        i += 1
    return limits


class UciEngine:
    def __init__(self, out: TextIO | None = None) -> None:
        self.out = out if out is not None else sys.stdout
        self._out_lock = threading.Lock()
        self._warm = threading.Event()
        self._warm_thread: threading.Thread | None = None
        self._search_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.board: Board | None = None
        self.searcher: Searcher | None = None
        self.move_overhead = 50
        self.limit_strength = False
        self.elo = MAX_ELO

    def send(self, line: str) -> None:
        with self._out_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def start_warmup(self) -> None:
        if self._warm_thread is None:
            self._warm_thread = threading.Thread(target=self._warmup, daemon=True)
            self._warm_thread.start()

    def _warmup(self) -> None:
        warmup()
        self.searcher = Searcher()
        self.board = Board()
        self._warm.set()

    def wait_ready(self) -> None:
        self.start_warmup()
        self._warm.wait()

    def stop_search(self) -> None:
        self._stop.set()
        self.wait_search()

    def wait_search(self) -> None:
        """Wait for the current search to finish by itself."""
        if self._search_thread is not None:
            self._search_thread.join()
            self._search_thread = None

    def handle(self, line: str) -> bool:
        """Process one command; returns False on ``quit``."""
        tokens = line.split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        if cmd == "uci":
            self.start_warmup()
            self.send(f"id name {ENGINE_NAME}")
            self.send(f"id author {AUTHOR}")
            self.send("option name Move Overhead type spin default 50 min 0 max 5000")
            self.send("option name UCI_LimitStrength type check default false")
            self.send(
                f"option name UCI_Elo type spin default {MAX_ELO} min {MIN_ELO} max {MAX_ELO}"
            )
            self.send("uciok")
        elif cmd == "isready":
            self.wait_ready()
            self.send("readyok")
        elif cmd == "setoption":
            self.set_option(args)
        elif cmd == "ucinewgame":
            self.stop_search()
            self.wait_ready()
            self.board = Board()
        elif cmd == "position":
            self.stop_search()
            self.wait_ready()
            self.set_position(args)
        elif cmd == "go":
            received = time.perf_counter()
            self.stop_search()
            self.wait_ready()
            if args[:1] == ["perft"]:
                self.perft(int(args[1]) if len(args) > 1 else 1)
            else:
                self.go(parse_go(args), received)
        elif cmd == "stop":
            self.stop_search()
        elif cmd == "quit":
            self.stop_search()
            return False
        elif cmd == "d":
            self.wait_ready()
            assert self.board is not None
            self.send(f"Fen: {self.board.fen()}")
        return True

    def set_option(self, args: list[str]) -> None:
        if "name" not in args:
            return
        rest = args[args.index("name") + 1 :]
        if "value" in rest:
            split = rest.index("value")
            name, value = " ".join(rest[:split]).lower(), " ".join(rest[split + 1 :])
        else:
            name, value = " ".join(rest).lower(), ""
        if name == "move overhead":
            self.move_overhead = max(0, int(value))
        elif name == "uci_limitstrength":
            self.limit_strength = value.lower() == "true"
        elif name == "uci_elo":
            self.elo = min(MAX_ELO, max(MIN_ELO, int(value)))

    def set_position(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "startpos":
            fen, rest = START_FEN, args[1:]
        elif args[0] == "fen":
            end = args.index("moves") if "moves" in args else len(args)
            fen, rest = " ".join(args[1:end]), args[end:]
        else:
            return
        board = Board(fen)
        for uci in rest[1:] if rest[:1] == ["moves"] else []:
            board.push_uci(uci)
        self.board = board

    def perft(self, depth: int) -> None:
        assert self.board is not None
        divide = self.board.divide(depth)
        for move, count in divide.items():
            self.send(f"{move}: {count}")
        self.send(f"\nNodes searched: {sum(divide.values())}")

    def go(self, limits: Limits, received: float | None = None) -> None:
        assert self.board is not None
        assert self.searcher is not None
        self._stop.clear()
        self.searcher.overhead_ms = self.move_overhead
        board, searcher = self.board, self.searcher

        def think() -> None:
            result = searcher.search(board, limits, self._on_info, self._stop, received)
            if limits.infinite:
                self._stop.wait()
            self._send_bestmove(result)

        self._search_thread = threading.Thread(target=think, daemon=True)
        self._search_thread.start()

    def _on_info(self, info: SearchInfo) -> None:
        self.send(format_info(info))

    def _send_bestmove(self, result: SearchResult) -> None:
        self.send(f"bestmove {move_to_uci(result.best_move) if result.best_move else '0000'}")


def main(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    engine = UciEngine(stdout)
    for line in stdin if stdin is not None else sys.stdin:
        if not engine.handle(line):
            break
    engine.stop_search()
    return 0
