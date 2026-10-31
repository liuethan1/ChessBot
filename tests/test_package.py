from __future__ import annotations

import hashlib
import io
import shlex
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import chessbot
from chessbot.tools import fetch
from chessbot.tools.common import split_command


def test_version() -> None:
    assert chessbot.__version__


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_split_command_round_trips_windows_paths(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    argv = [r"C:\Program Files\Python\python.exe", "-m", "chessbot"]
    assert split_command(shlex.join(argv)) == argv
    assert split_command("python -m chessbot") == ["python", "-m", "chessbot"]


def test_split_command_windows_double_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    command = r'"C:\Program Files\Stockfish\stockfish.exe" --flag'
    assert split_command(command) == [r"C:\Program Files\Stockfish\stockfish.exe", "--flag"]


def test_numba_jit_compiles() -> None:
    import numba
    import numpy as np

    @numba.njit(cache=False)
    def popcount_sum(values: np.ndarray) -> int:
        total = 0
        for v in values:
            while v:
                v &= v - np.uint64(1)
                total += 1
        return total

    assert popcount_sum(np.array([0xFF, 0x0F00], dtype=np.uint64)) == 12


def test_every_asset_is_pinned() -> None:
    for assets in fetch.ASSETS.values():
        for asset in assets:
            assert len(asset.sha256) == 64
            assert "/master/" not in asset.url
            assert "/main/" not in asset.url


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(payload))


def test_fetch_extracts_zip_and_verifies_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("book.epd", "fen\n")
    payload = buffer.getvalue()
    _serve(monkeypatch, payload)

    good = fetch.Asset("u", hashlib.sha256(payload).hexdigest(), "books/book.epd", "book.epd")
    assert fetch.fetch(good, tmp_path).read_text() == "fen\n"
    bad = fetch.Asset("u", "0" * 64, "books/other.epd", "book.epd")
    with pytest.raises(ValueError, match="checksum"):
        fetch.fetch(bad, tmp_path)


def test_fetch_extracts_tar_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("dir/tool")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"\x7fELF"))
    payload = buffer.getvalue()
    _serve(monkeypatch, payload)

    asset = fetch.Asset("u", hashlib.sha256(payload).hexdigest(), "bin/tool", "dir/tool")
    target = fetch.fetch(asset, tmp_path)
    assert target.read_bytes() == b"\x7fELF"
    if sys.platform != "win32":
        assert target.stat().st_mode & 0o111
