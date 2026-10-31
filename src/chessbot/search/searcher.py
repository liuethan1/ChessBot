"""Iterative deepening driver around the jitted search, with time management."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from chessbot.core.board import Board
from chessbot.search.evaluate import EvalTables, eval_tables
from chessbot.search.negamax import (
    INF,
    MATE,
    MATE_BOUND,
    SC_NODES,
    SC_PREV_PV_LEN,
    SC_SELDEPTH,
    SC_VALUE,
    SearchStack,
    new_stack,
    run,
    start_iteration,
    unwind,
)
from chessbot.search.tt import DEFAULT_HASH_MB, TransTable, clear_tt, new_search_generation, new_tt

MAX_DEPTH = 100
CHUNK_NODES = 2048
"""Nodes searched per call into the jitted code; the clock and ``stop`` are checked in between."""
DEFAULT_MOVES_TO_GO = 30
SOFT_STOP_FRACTION = 0.55
"""Do not start another iteration once this fraction of the soft budget is used."""
ASPIRATION_MIN_DEPTH = 4
ASPIRATION_DELTA = 20
"""Half-width in centipawns of the first window around the previous iteration's score."""
ASPIRATION_MAX_DELTA = 600
"""Past this half-width the window is opened fully."""


@dataclass
class Limits:
    """A UCI ``go`` command. Times are in milliseconds."""

    wtime: int | None = None
    btime: int | None = None
    winc: int = 0
    binc: int = 0
    movestogo: int | None = None
    movetime: int | None = None
    depth: int | None = None
    nodes: int | None = None
    infinite: bool = False


@dataclass
class TimeBudget:
    soft_ms: float | None
    """Target thinking time; no new iteration starts after ``SOFT_STOP_FRACTION`` of it."""
    hard_ms: float | None
    """The search is abandoned mid-iteration at this point."""


def plan_time(limits: Limits, white_to_move: bool, overhead_ms: int) -> TimeBudget:
    if limits.infinite:
        return TimeBudget(None, None)
    if limits.movetime is not None:
        budget = max(1.0, limits.movetime - overhead_ms)
        return TimeBudget(budget, budget)
    remaining = limits.wtime if white_to_move else limits.btime
    if remaining is None:
        return TimeBudget(None, None)
    inc = limits.winc if white_to_move else limits.binc
    usable = max(1.0, remaining - overhead_ms)
    moves_to_go = min(limits.movestogo or DEFAULT_MOVES_TO_GO, DEFAULT_MOVES_TO_GO)
    soft = usable / moves_to_go + 0.75 * inc
    # With 30% of the remaining time as the cap, a long game at 10+0.1 settles at a clock of
    # about 200 ms instead of draining towards the overhead.
    hard = min(soft * 4, usable * 0.3 + inc * 0.5)
    soft = min(soft, hard)
    return TimeBudget(max(1.0, soft), max(1.0, hard))


@dataclass
class SearchInfo:
    depth: int
    seldepth: int
    score: int
    nodes: int
    seconds: float
    pv: list[int] = field(default_factory=list)

    @property
    def mate_in(self) -> int | None:
        """Moves to mate (negative if the side to move is mated), or None for a normal score."""
        if self.score > MATE_BOUND:
            return (MATE - self.score + 1) // 2
        if self.score < -MATE_BOUND:
            return -((MATE + self.score) // 2)
        return None


@dataclass
class SearchResult:
    best_move: int
    """0 when there is no legal move."""
    score: int
    depth: int
    nodes: int
    seconds: float
    pv: list[int]


InfoCallback = Callable[[SearchInfo], None]


class Searcher:
    def __init__(self, ev: EvalTables | None = None, hash_mb: int = DEFAULT_HASH_MB) -> None:
        self.ev = ev if ev is not None else eval_tables()
        self.ss: SearchStack = new_stack()
        self.tt: TransTable = new_tt(hash_mb)
        self.overhead_ms = 50

    def resize_hash(self, mb: int) -> None:
        self.tt = new_tt(mb)

    def new_game(self) -> None:
        """Forget everything learned in earlier searches (``ucinewgame``)."""
        clear_tt(self.tt)
        self.ss = new_stack()

    def search(
        self,
        board: Board,
        limits: Limits,
        on_info: InfoCallback | None = None,
        stop: threading.Event | None = None,
        start: float | None = None,
    ) -> SearchResult:
        """Search ``board`` within ``limits``; ``start`` is when the clock started (``go``
        arrived), as a ``time.perf_counter()`` value, and defaults to now."""
        if start is None:
            start = time.perf_counter()
        budget = plan_time(limits, board.white_to_move, self.overhead_ms)
        timed = budget.hard_ms is not None
        max_depth = min(limits.depth or MAX_DEPTH, MAX_DEPTH)
        legal = board.legal_moves()
        if not legal:
            return SearchResult(0, 0, 0, 0, 0.0, [])

        ss, pos, t, ev, tt = self.ss, board.pos, board.t, self.ev, self.tt
        new_search_generation(tt)
        ss.killers.fill(0)
        ss.sc[SC_NODES] = 0
        ss.sc[SC_PREV_PV_LEN] = 0
        result = SearchResult(legal[0], 0, 0, 0, 0.0, [legal[0]])

        def elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000

        def run_window(depth: int, alpha: int, beta: int) -> bool:
            """Search the root to ``depth`` within ``(alpha, beta)``; False if abandoned."""
            start_iteration(ss, depth, alpha, beta)
            while True:
                chunk = CHUNK_NODES
                if limits.nodes is not None:
                    chunk = min(chunk, limits.nodes - int(ss.sc[SC_NODES]))
                    if chunk <= 0:
                        return False
                if run(pos, t, ev, ss, tt, chunk):
                    return True
                if stop is not None and stop.is_set():
                    return False
                if budget.hard_ms is not None and depth > 1 and elapsed_ms() >= budget.hard_ms:
                    return False

        def root_pv() -> list[int]:
            pv_len = int(ss.pv_len[0])
            return [int(m) for m in ss.pv[0, :pv_len]]

        score = 0
        for depth in range(1, max_depth + 1):
            delta = ASPIRATION_DELTA
            alpha, beta = -INF, INF
            if depth >= ASPIRATION_MIN_DEPTH and abs(score) < MATE_BOUND:
                alpha, beta = max(score - delta, -INF), min(score + delta, INF)
            while True:
                finished = run_window(depth, alpha, beta)
                if not finished:
                    break
                value = int(ss.sc[SC_VALUE])
                pv = root_pv()
                if pv:
                    ss.prev_pv[: len(pv)] = pv
                    ss.sc[SC_PREV_PV_LEN] = len(pv)
                if value <= alpha:
                    beta = (alpha + beta) // 2
                    alpha = max(value - delta, -INF)
                elif value >= beta:
                    beta = min(value + delta, INF)
                    result.best_move, result.pv, result.score = pv[0], pv, value
                else:
                    break
                delta += delta // 2
                if delta > ASPIRATION_MAX_DELTA:
                    alpha, beta = -INF, INF

            nodes = int(ss.sc[SC_NODES])
            if not finished:
                unwind(pos, t, ss)
                # A root move that beat alpha in the abandoned iteration was fully searched
                # (PVS re-searches before it is accepted), so it is at least as good as the last
                # completed iteration's choice.
                pv = root_pv()
                if pv:
                    result.best_move, result.pv, result.score = pv[0], pv, int(ss.best[0])
                result.nodes = nodes
                break

            score = value
            result = SearchResult(pv[0], score, depth, nodes, 0.0, pv)
            info = SearchInfo(depth, int(ss.sc[SC_SELDEPTH]), score, nodes, elapsed_ms() / 1000, pv)
            if on_info is not None:
                on_info(info)

            if stop is not None and stop.is_set():
                break
            if limits.nodes is not None and nodes >= limits.nodes:
                break
            if timed:
                if len(legal) == 1:
                    break
                if abs(score) > MATE_BOUND and MATE - abs(score) <= depth:
                    break
                assert budget.soft_ms is not None
                if elapsed_ms() >= budget.soft_ms * SOFT_STOP_FRACTION:
                    break

        result.seconds = elapsed_ms() / 1000
        return result


def warmup_search() -> None:
    """Compile (or load from the cache) the search entry points with a tiny search."""
    board = Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    Searcher(hash_mb=1).search(board, Limits(depth=3))
    Searcher(hash_mb=1).search(board, Limits(nodes=100))
