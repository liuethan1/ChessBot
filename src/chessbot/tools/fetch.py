"""Download pinned, checksummed external assets into ``external/``.

* ``books``: 8moves_v3 (balanced, for weaker engines) and UHO_4060_v4 (unbalanced, for SPRT once
  the engine is strong), from official-stockfish/books.
* ``suites``: WAC (300 tactical positions) and STS 1-15 (1500 positional positions).
* ``fastchess``: the fastchess match runner (Linux x86-64 release build).
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
from dataclasses import dataclass
from pathlib import Path

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
}


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
    with urllib.request.urlopen(asset.url, timeout=120) as response:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", nargs="+", choices=[*ASSETS, "all"])
    parser.add_argument("--dest", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    groups = list(ASSETS) if "all" in args.what else args.what
    if "fastchess" in groups and (sys.platform != "linux" or platform.machine() != "x86_64"):
        print("fastchess: only the Linux x86-64 build is pinned; install it manually from")
        print("  https://github.com/Disservin/fastchess/releases and set FASTCHESS_PATH")
        groups.remove("fastchess")
    for group in groups:
        for asset in ASSETS[group]:
            print(f"{group}: {fetch(asset, args.dest, args.force)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
