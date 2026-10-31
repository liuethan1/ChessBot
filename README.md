# ChessBot

A rewrite of the original Keras move-predictor (now in [`legacy/`](legacy/)) as a real chess
engine: a UCI engine written in Python with numba/numpy speedups, alpha-beta search, and later an
NNUE evaluation, an opening book, and Syzygy endgame tablebases.

The engine is built in phases (see the remake plan). Phase 0 adds the package skeleton and the
measurement tooling used to judge every later change. Phase 1 adds the engine core in
`chessbot.core`: numba-jitted bitboards with magic sliders, FEN I/O, Zobrist keys, make/unmake,
and a legal move generator that passes both perft suites. Phase 2 makes it a playable UCI engine:
`chessbot.search` adds iterative deepening alpha-beta with quiescence, a PeSTO tapered evaluation,
MVV-LVA capture ordering, draw detection, and time management.

## Playing

```bash
python -m chessbot          # or the `chessbot` script; speaks UCI on stdin/stdout
```

Point a GUI, fastchess, cutechess-cli, or lichess-bot at that command. Supported: `go` with
`wtime`/`btime`/`winc`/`binc`/`movestogo`/`movetime`/`depth`/`nodes`/`infinite`, `stop`,
`go perft N`, `bench [depth]` (searches the bench positions from an empty table and prints the
node count and speed; default depth 6), and `d` (prints the FEN). Options: `Hash` (transposition
table size in MB, default 16), `Move Overhead` (ms kept in reserve per move, default 50), plus
`UCI_LimitStrength` and `UCI_Elo`, which are accepted but have no effect yet. Arguments run one
command and exit, as in `python -m chessbot bench 6`.

## Search

The search (`chessbot/search/negamax.py`) is a fail-soft negamax over an explicit stack rather
than a recursive function, because numba segfaults when it reloads a cached recursive function.
That also makes it resumable: the Python driver (`searcher.py`) runs it in chunks of 2048 nodes
and checks the clock and `stop` in between, so the jitted code needs no clock. Nodes at depth 0
or below are quiescence nodes (stand pat, then captures and queen promotions that do not lose
material by static exchange evaluation, or every evasion when in check). Repetitions, the 50-move
rule, and bare-minor endings score as draws.

- **Transposition table** (`tt.py`): 16-byte entries (key, move, score, static eval, depth, bound,
  generation) in parallel numpy arrays, indexed by the low key bits. Every completed frame is
  stored; null-window frames return straight from an entry that is deep enough and whose bound
  proves the result. Mate scores are stored relative to the node.
- **Move ordering**: previous-iteration PV move, table move, captures that win or break even by SEE
  (MVV-LVA among them), two killers per ply, the counter-move to the opponent's last move, quiet
  moves by history (with gravity, and a penalty for quiets that failed before a cutoff), losing
  captures, underpromotions.
- **Principal variation search**: moves after the first are searched with a null window and
  searched again with the full window only if they land inside it. The re-search is a stage of
  the parent frame, so the search stays non-recursive.
- **Aspiration windows**: from depth 4 each iteration starts at +/-20 cp around the previous score
  and widens by half on each fail low or fail high.
- **Check extension**: a side in check at a full-width node is searched one ply deeper.

Time management: with a clock, the target is `remaining / 30 + 0.75 * increment` (fewer moves if
`movestogo` is smaller); no new iteration starts after 55% of the target, and the search is cut
off mid-iteration at `min(4 * target, 0.3 * remaining + increment / 2)`. `Move Overhead` is
subtracted from the remaining time first.

## Engine core and numba compile time

Every jitted function uses `cache=True`. The first run on a machine compiles the core and the
search (about 10-20 seconds); later processes load it from the cache
(`~/.cache/chessbot/numba-<hash>`, override with `CHESSBOT_CACHE_DIR`). The cache directory is
keyed by a hash of the `chessbot.core` and `chessbot.search` sources, so editing any of them
triggers a clean recompile. `chessbot.search.warmup()` loads everything; the UCI loop starts it on
`uci` and waits for it before answering `isready`, so it never runs on the clock.

```bash
chessbot-warmup --check 5   # fails if a fresh process needs more than 5 s to be ready
chessbot-perft --suite standard --all-depths --backend py:chessbot.core.perft:perft_fen
```

```python
from chessbot.core import Board

board = Board()
board.push_uci("e2e4")
board.legal_uci(), board.fen(), board.key, board.perft(5)
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
chessbot-fetch all          # opening books, WAC/STS suites, and fastchess into external/
```

Stockfish is needed for ACPL analysis, reference gauntlets, and some tests. Install it from your
package manager (`apt install stockfish`) or set `STOCKFISH_PATH`.

## Measurement tools

| Command | What it does |
|---|---|
| `chessbot-perft` | Perft against the CPW or standard suite; on a mismatch, walks the tree to the first wrong move list. Backends: `reference` (python-chess), `uci:CMD` (`go perft`), `py:MODULE:FUNC` |
| `chessbot-bench` | Searches a fixed set of positions at a fixed depth through UCI; prints total nodes (a per-commit fingerprint) and nodes per second |
| `chessbot-suite` | Runs an EPD test suite (WAC, STS, ...) at a fixed time, depth, or node limit and reports the solve count (and STS points) |
| `chessbot-match` | Runs an SPRT or fixed-length match with fastchess (default) or cutechess-cli on an opening book |
| `chessbot-sprt` | Computes the SPRT log-likelihood ratio and an Elo estimate from W/D/L or pentanomial counts |
| `chessbot-acpl` | Average centipawn loss, blunders, and mistakes for each side of a PGN, using Stockfish |
| `chessbot-warmup` | Times how long a fresh process takes to import and warm up the jitted core and search; `--check SECONDS` fails CI when the numba cache is not doing its job |
| `chessbot-fetch` | Downloads pinned, checksummed books, suites, and the fastchess binary into `external/` |

Examples:

```bash
chessbot-perft --suite standard --backend "uci:stockfish"
chessbot-bench --engine stockfish --depth 10
chessbot-bench --engine "python -m chessbot" --depth 5
chessbot-suite external/suites/wac.epd --engine stockfish --movetime 0.1
chessbot-match --new "./engine-new" --base "./engine-main" --tc 8+0.08 --sprt 0 5 --concurrency 4
chessbot-match --new "./engine-new" --base stockfish --base-option UCI_LimitStrength=true --base-option UCI_Elo=1500 --games 200
chessbot-acpl legacy/legacy-selfplay-game.pgn --depth 14
```

## Development

```bash
ruff check . && ruff format --check . && mypy && pytest
```

Tests that need Stockfish or fastchess are skipped when the binary is not found. Tests marked
`slow` (such as reproducing the legacy bot's ACPL baseline) run in a separate CI job.
