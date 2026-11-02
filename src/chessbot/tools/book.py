"""Build a Polyglot opening book from PGN (or zipped PGN) of strong games."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import chess.polyglot

from chessbot.book import DEFAULT_BOOK_DEPTH, encode_polyglot_move

ENTRY_STRUCT = chess.polyglot.ENTRY_STRUCT


def _read_games(handle: TextIO) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game


def iter_games(paths: list[Path]) -> Iterator[chess.pgn.Game]:
    for path in paths:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if Path(name).suffix.lower() != ".pgn":
                        continue
                    with archive.open(name) as raw:
                        handle: TextIO = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                        yield from _read_games(handle)
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            yield from _read_games(handle)


def _elo(headers: chess.pgn.Headers, key: str) -> int:
    raw = headers.get(key, "")
    try:
        return int(raw)
    except ValueError:
        return 0


def include_game(game: chess.pgn.Game, min_rating: int) -> bool:
    headers = game.headers
    if headers.get("Variant", "Standard") not in {"", "Standard", "standard"}:
        return False
    weakest = min(_elo(headers, "WhiteElo"), _elo(headers, "BlackElo"))
    return min_rating <= 0 or weakest >= min_rating


def collect_entries(
    paths: list[Path],
    *,
    max_ply: int = DEFAULT_BOOK_DEPTH,
    min_rating: int = 0,
) -> dict[int, dict[int, int]]:
    """Map Polyglot key -> {raw_move: count} from every included game."""
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    games = 0
    used = 0
    for game in iter_games(paths):
        games += 1
        if games % 20000 == 0:
            print(f"  scanned {games} games, kept {used}", flush=True)
        if not include_game(game, min_rating):
            continue
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= max_ply:
                break
            if not board.is_legal(move):
                break
            key = chess.polyglot.zobrist_hash(board)
            raw = encode_polyglot_move(board, move)
            counts[key][raw] += 1
            board.push(move)
        used += 1
    print(f"scanned {games} games, kept {used}, {len(counts)} positions", flush=True)
    return counts


def write_book(
    counts: dict[int, dict[int, int]],
    dest: Path,
    *,
    min_games: int = 8,
) -> int:
    """Write a sorted Polyglot file. Returns the number of entries kept."""
    records: list[bytes] = []
    for key, moves in counts.items():
        kept = {raw: n for raw, n in moves.items() if n >= min_games}
        if not kept:
            continue
        peak = max(kept.values())
        scale = 1 if peak <= 65535 else max(1, (peak + 65534) // 65535)
        for raw, n in kept.items():
            weight = max(1, min(65535, n // scale))
            records.append(ENTRY_STRUCT.pack(key, raw, weight, 0))
    records.sort()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(records))
    return len(records)


def book_info(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    if size % ENTRY_STRUCT.size:
        raise ValueError(f"{path} is not a Polyglot book")
    n = size // ENTRY_STRUCT.size
    with chess.polyglot.open_reader(path) as reader:
        keys = len({entry.key for entry in reader})
    return n, keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="build a .bin book from PGN/zip files")
    build.add_argument("pgn", nargs="+", type=Path)
    build.add_argument("-o", "--output", type=Path, required=True)
    build.add_argument("--max-ply", type=int, default=DEFAULT_BOOK_DEPTH)
    build.add_argument("--min-games", type=int, default=8)
    build.add_argument(
        "--min-rating",
        type=int,
        default=0,
        help="both players must be at least this Elo (0 = keep every tagged game)",
    )

    info = sub.add_parser("info", help="print entry and position counts")
    info.add_argument("book", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "info":
        entries, keys = book_info(args.book)
        print(f"{args.book}: {entries} entries, {keys} positions")
        return 0

    counts = collect_entries(args.pgn, max_ply=args.max_ply, min_rating=args.min_rating)
    n = write_book(counts, args.output, min_games=args.min_games)
    print(f"wrote {n} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
