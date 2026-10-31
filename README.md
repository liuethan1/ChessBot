# ChessBot

A rewrite of the original Keras move-predictor (now in [`legacy/`](legacy/)) as a real chess
engine: a UCI engine written in Python with numba/numpy speedups, alpha-beta search, and later an
NNUE evaluation, an opening book, and Syzygy endgame tablebases.

The engine is built in phases (see the remake plan). Phase 0 adds the package skeleton and the
measurement tooling used to judge every later change.

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
| `chessbot-fetch` | Downloads pinned, checksummed books, suites, and the fastchess binary into `external/` |

Examples:

```bash
chessbot-perft --suite standard --backend "uci:stockfish"
chessbot-bench --engine stockfish --depth 10
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
