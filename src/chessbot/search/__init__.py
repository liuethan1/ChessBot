"""Search and evaluation: iterative deepening alpha-beta with quiescence and Texel-tuned HCE."""

from __future__ import annotations

import time

from chessbot.search.searcher import Limits, Searcher, SearchInfo, SearchResult, warmup_search


def warmup() -> float:
    """Load every jitted function of the engine (core and search); returns seconds taken."""
    from chessbot.core import warmup as core_warmup

    start = time.perf_counter()
    core_warmup()
    warmup_search()
    return time.perf_counter() - start


__all__ = ["Limits", "SearchInfo", "SearchResult", "Searcher", "warmup"]
