"""Download pinned, checksummed external assets into ``external/``.

* ``books``: 8moves_v3 (balanced, for weaker engines) and UHO_4060_v4 (unbalanced, for SPRT once
  the engine is strong), from official-stockfish/books.
* ``suites``: WAC (300 tactical positions) and STS 1-15 (1500 positional positions).
* ``fastchess``: the fastchess match runner (Linux x86-64 release build).
* ``elite``: recent Lichess Elite PGN zips (2500+ vs 2300+, no bullet) for building the Polyglot
  book. Not part of ``all``.
* ``syzygy``: 3-5 piece WDL/DTZ tables from tablebase.lichess.ovh (~1 GB). Not part of ``all``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import chess.syzygy

from chessbot.tools.common import EXTERNAL_DIR

BOOKS_REV = "a804a97e709f40aa75a00f9d19e84915b7a3a45c"
WAC_REV = "29c2e6481e6f9b35adf3ec136e7f69150a102e99"
STS_REV = "4e36977090c3a93ea57d5f34e13e42df422634d3"
FASTCHESS_VERSION = "v1.8.2-alpha"


@dataclass(frozen=True)
class Asset:
    url: str
    sha256: str
    dest: str
    """Path under external/ of the file to produce."""
    member: str | None = None
    """File to extract when the download is a zip or tar archive."""


ASSETS: dict[str, list[Asset]] = {
    "books": [
        Asset(
            f"https://raw.githubusercontent.com/official-stockfish/books/{BOOKS_REV}/8moves_v3.pgn.zip",
            "7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e",
            "books/8moves_v3.pgn",
            member="8moves_v3.pgn",
        ),
        Asset(
            f"https://raw.githubusercontent.com/official-stockfish/books/{BOOKS_REV}/UHO_4060_v4.epd.zip",
            "a97424c5b98b42f8c27ff450f0681ad11696148548c975752350e98417ead11d",
            "books/UHO_4060_v4.epd",
            member="UHO_4060_v4.epd",
        ),
    ],
    "suites": [
        Asset(
            f"https://raw.githubusercontent.com/akanalytics/odonata/{WAC_REV}/crates/odonata-base/resources/wac.epd",
            "b4bf0fd3b76fca52fb490d446d23c3f6435801e6f2e2d127ddf06bedd2a969b3",
            "suites/wac.epd",
        ),
        Asset(
            f"https://raw.githubusercontent.com/fsmosca/STS-Rating/{STS_REV}/STS1-STS15_LAN_v3.epd",
            "d7a33ae6cf2fb5f3f18ef1b904cca07dce2cd0458edcf60644f59d4d9e28c07b",
            "suites/sts.epd",
        ),
    ],
    "fastchess": [
        Asset(
            f"https://github.com/Disservin/fastchess/releases/download/{FASTCHESS_VERSION}/fastchess-linux-x86-64.tar",
            "1003f920bebe841acdab5e6d7871e93171a4586857b3ec91c21ef5777f26ff96",
            "bin/fastchess",
            member="fastchess-linux-x86-64/fastchess",
        ),
    ],
    "elite": [
        Asset(
            "https://database.nikonoel.fr/lichess_elite_2025-07.zip",
            "359ede2617548a3dbb1c780549fdffab1f59cf174d4f8bb4f882b190847f3f4b",
            "elite/lichess_elite_2025-07.zip",
        ),
        Asset(
            "https://database.nikonoel.fr/lichess_elite_2025-08.zip",
            "06a40ce151547b6a7322ce7407c47581e5d58baf83ae4960b5958bcf4a45dbd3",
            "elite/lichess_elite_2025-08.zip",
        ),
        Asset(
            "https://database.nikonoel.fr/lichess_elite_2025-09.zip",
            "81e554084af620c1b5822b7d776f4f06b4dee5a50c68727df912329af2a51b48",
            "elite/lichess_elite_2025-09.zip",
        ),
    ],
}

CORE_GROUPS = ("books", "suites", "fastchess")
SYZYGY_WDL = "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl"
SYZYGY_DTZ = "https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz"
SYZYGY_SHA256 = "https://tablebase.lichess.ovh/tables/standard/sha256"
RTBW_MAGIC = b"q\xe8#]"
RTBZ_MAGIC = b"\xd7f\x0c\xa5"
SYZYGY_WORKERS = 8


def _extract(payload: bytes, member: str) -> bytes:
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return archive.read(member)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        return extracted.read()


def fetch(asset: Asset, root: Path, force: bool = False) -> Path:
    target = root / asset.dest
    if target.exists() and not force:
        return target
    with urllib.request.urlopen(asset.url, timeout=300) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != asset.sha256:
        raise ValueError(f"checksum mismatch for {asset.url}: {digest}")
    data = _extract(payload, asset.member) if asset.member else payload
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    shutil.move(tmp, target)
    if target.parent.name == "bin":
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return bytes(response.read())


def syzygy_names(max_pieces: int = 5) -> list[str]:
    names = sorted({name for name in chess.syzygy.tablenames(piece_count=max_pieces)})
    return [name for name in names if len(name) - 1 <= max_pieces]


def fetch_syzygy(root: Path, *, max_pieces: int = 5, force: bool = False) -> Path:
    """Download 3-``max_pieces`` Syzygy WDL/DTZ files into ``root/syzygy``."""
    dest = root / "syzygy"
    dest.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for line in _download(SYZYGY_SHA256).decode().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            checksums[parts[-1]] = parts[0]
    jobs: list[tuple[str, Path, bytes]] = []
    for name in syzygy_names(max_pieces):
        jobs.append((f"{SYZYGY_WDL}/{name}.rtbw", dest / f"{name}.rtbw", RTBW_MAGIC))
        jobs.append((f"{SYZYGY_DTZ}/{name}.rtbz", dest / f"{name}.rtbz", RTBZ_MAGIC))

    def one(url: str, target: Path, magic: bytes) -> Path:
        if target.exists() and not force:
            return target
        payload = _download(url)
        if not payload.startswith(magic):
            raise ValueError(f"bad magic for {url}")
        expected = checksums.get(target.name)
        if expected is not None:
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected:
                raise ValueError(f"checksum mismatch for {url}: {digest}")
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(payload)
        shutil.move(tmp, target)
        return target

    done = 0
    with ThreadPoolExecutor(max_workers=SYZYGY_WORKERS) as pool:
        futures = [pool.submit(one, url, target, magic) for url, target, magic in jobs]
        for future in as_completed(futures):
            path = future.result()
            done += 1
            if done == 1 or done % 40 == 0 or done == len(jobs):
                print(f"syzygy: {done}/{len(jobs)} {path.name}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", nargs="+", choices=[*ASSETS, "all", "syzygy"])
    parser.add_argument("--dest", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=5,
        choices=(3, 4, 5),
        help="Syzygy piece limit (default 5, about 1 GB)",
    )
    args = parser.parse_args(argv)

    groups = list(CORE_GROUPS) if "all" in args.what else list(args.what)
    if "fastchess" in groups and (sys.platform != "linux" or platform.machine() != "x86_64"):
        print("fastchess: only the Linux x86-64 build is pinned; install it manually from")
        print("  https://github.com/Disservin/fastchess/releases and set FASTCHESS_PATH")
        groups.remove("fastchess")
    for group in groups:
        if group == "syzygy":
            print(
                f"syzygy: {fetch_syzygy(args.dest, max_pieces=args.max_pieces, force=args.force)}"
            )
            continue
        for asset in ASSETS[group]:
            print(f"{group}: {fetch(asset, args.dest, args.force)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
