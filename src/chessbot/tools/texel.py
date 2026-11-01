"""Texel-tune extra evaluation terms on quiet positions labelled with game results.

The method (Peter Österlund) minimises the mean squared error between the game result and a
logistic of the static eval. Positions are taken from PGNs after the opening, skipping checks and
captures so the eval is not asked to see tactics. Feature extraction uses the engine's own jitted
:func:`eval_components`; python-chess is only used to parse the PGN.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt

from chessbot.core.board import Board
from chessbot.search.evaluate import (
    I_TEMPO,
    MAX_PHASE,
    N_TERMS,
    TERM_NAMES,
    W_EG,
    W_MG,
    EvalTables,
    eval_components,
    eval_tables,
)

WDL = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _taper_vec(
    pst_mg: npt.NDArray[np.float64],
    pst_eg: npt.NDArray[np.float64],
    phase: npt.NDArray[np.float64],
    feat: npt.NDArray[np.float64],
    w_mg: npt.NDArray[np.float64],
    w_eg: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    ph = np.clip(phase, 0, MAX_PHASE)
    mg = pst_mg + feat @ w_mg
    eg = pst_eg + feat @ w_eg
    return (mg * ph + eg * (MAX_PHASE - ph)) / MAX_PHASE  # type: ignore[no-any-return]


def sigmoid(score: npt.NDArray[np.float64], k: float) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-score / k))  # type: ignore[no-any-return]


def mse(
    data: dict[str, npt.NDArray[np.float64]],
    w_mg: npt.NDArray[np.float64],
    w_eg: npt.NDArray[np.float64],
    k: float,
) -> float:
    p = sigmoid(
        _taper_vec(data["pst_mg"], data["pst_eg"], data["phase"], data["feat"], w_mg, w_eg), k
    )
    return float(np.mean((p - data["result"]) ** 2))


def extract_game(
    game: chess.pgn.Game,
    min_ply: int = 16,
    tables: EvalTables | None = None,
) -> tuple[list[npt.NDArray[np.int32]], list[int], list[int], list[int], list[float]]:
    """Quiet positions of one game: features from the side to move, plus PeSTO components."""
    result = WDL.get(game.headers.get("Result", ""))
    if result is None:
        return [], [], [], [], []
    ev = tables if tables is not None else eval_tables()
    board = game.board()
    feats: list[npt.NDArray[np.int32]] = []
    pst_mg: list[int] = []
    pst_eg: list[int] = []
    phase: list[int] = []
    labels: list[float] = []
    feat = np.zeros(N_TERMS, np.int32)
    ply = 0
    for move in game.mainline_moves():
        capture = board.is_capture(move)
        promo = move.promotion is not None
        board.push(move)
        ply += 1
        if ply < min_ply or capture or promo or board.is_check() or board.is_game_over():
            continue
        pos = Board(board.fen())
        mg, eg, ph = eval_components(pos.pos, pos.t, ev, feat)
        row = feat.astype(np.int32, copy=True)
        if not pos.white_to_move:
            row *= -1
            mg, eg = -mg, -eg
        row[I_TEMPO] = 1
        feats.append(row)
        pst_mg.append(mg)
        pst_eg.append(eg)
        phase.append(ph)
        labels.append(result if pos.white_to_move else 1.0 - result)
    return feats, pst_mg, pst_eg, phase, labels


def extract_pgns(
    paths: Sequence[Path], min_ply: int = 16, max_positions: int | None = None
) -> dict[str, npt.NDArray[np.float64]]:
    ev = eval_tables()
    feats: list[npt.NDArray[np.int32]] = []
    pst_mg: list[int] = []
    pst_eg: list[int] = []
    phase: list[int] = []
    labels: list[float] = []
    for path in paths:
        with path.open() as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                f, mg, eg, ph, y = extract_game(game, min_ply, ev)
                feats.extend(f)
                pst_mg.extend(mg)
                pst_eg.extend(eg)
                phase.extend(ph)
                labels.extend(y)
                if max_positions is not None and len(labels) >= max_positions:
                    feats = feats[:max_positions]
                    pst_mg = pst_mg[:max_positions]
                    pst_eg = pst_eg[:max_positions]
                    phase = phase[:max_positions]
                    labels = labels[:max_positions]
                    break
        if max_positions is not None and len(labels) >= max_positions:
            break
    if not labels:
        raise ValueError("no quiet labelled positions found")
    return {
        "feat": np.asarray(feats, np.float64),
        "pst_mg": np.asarray(pst_mg, np.float64),
        "pst_eg": np.asarray(pst_eg, np.float64),
        "phase": np.asarray(phase, np.float64),
        "result": np.asarray(labels, np.float64),
    }


def adam_tune(
    data: dict[str, npt.NDArray[np.float64]],
    *,
    epochs: int = 40,
    lr: float = 0.05,
    k: float = 110.0,
    l2: float = 2e-4,
    batch: int = 4096,
    seed: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Tune extra-term weights; returns (w_mg, w_eg, final MSE). PeSTO PST values stay frozen."""
    rng = np.random.default_rng(seed)
    n = data["result"].shape[0]
    w_mg = np.array(W_MG, np.float64)
    w_eg = np.array(W_EG, np.float64)
    w0_mg, w0_eg = w_mg.copy(), w_eg.copy()
    m_mg = np.zeros(N_TERMS)
    v_mg = np.zeros(N_TERMS)
    m_eg = np.zeros(N_TERMS)
    v_eg = np.zeros(N_TERMS)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            feat = data["feat"][idx]
            pst_mg = data["pst_mg"][idx]
            pst_eg = data["pst_eg"][idx]
            phase = np.clip(data["phase"][idx], 0, MAX_PHASE)
            y = data["result"][idx]
            score = _taper_vec(pst_mg, pst_eg, phase, feat, w_mg, w_eg)
            p = sigmoid(score, k)
            # d MSE / d score = 2 (p - y) p (1-p) / K
            ds = 2.0 * (p - y) * p * (1.0 - p) / k
            mg_scale = phase / MAX_PHASE
            eg_scale = (MAX_PHASE - phase) / MAX_PHASE
            g_mg = (feat * (ds * mg_scale)[:, None]).mean(axis=0) + l2 * (w_mg - w0_mg)
            g_eg = (feat * (ds * eg_scale)[:, None]).mean(axis=0) + l2 * (w_eg - w0_eg)
            step += 1
            m_mg = beta1 * m_mg + (1 - beta1) * g_mg
            v_mg = beta2 * v_mg + (1 - beta2) * g_mg * g_mg
            m_eg = beta1 * m_eg + (1 - beta1) * g_eg
            v_eg = beta2 * v_eg + (1 - beta2) * g_eg * g_eg
            mh_mg = m_mg / (1 - beta1**step)
            vh_mg = v_mg / (1 - beta2**step)
            mh_eg = m_eg / (1 - beta1**step)
            vh_eg = v_eg / (1 - beta2**step)
            w_mg -= lr * mh_mg / (np.sqrt(vh_mg) + eps)
            w_eg -= lr * mh_eg / (np.sqrt(vh_eg) + eps)
    return w_mg, w_eg, mse(data, w_mg, w_eg, k)


def format_weights(
    w_mg: Sequence[float] | npt.NDArray[np.float64],
    w_eg: Sequence[float] | npt.NDArray[np.float64],
) -> str:
    def tup(values: Sequence[float] | npt.NDArray[np.float64]) -> str:
        body = ", ".join(str(round(float(v))) for v in values)
        return f"({body})"

    lines = ["W_MG = " + tup(w_mg), "W_EG = " + tup(w_eg)]
    for i, name in enumerate(TERM_NAMES):
        lines.append(f"# {name}: {round(float(w_mg[i]))} / {round(float(w_eg[i]))}")
    return "\n".join(lines) + "\n"


def _cmd_extract(args: argparse.Namespace) -> int:
    data = extract_pgns([Path(p) for p in args.pgn], args.min_ply, args.max_positions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"wrote {data['result'].shape[0]} positions to {args.output}", flush=True)
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    loaded = np.load(args.data)
    keys = ("feat", "pst_mg", "pst_eg", "phase", "result")
    data = {k: loaded[k].astype(np.float64) for k in keys}
    n = data["result"].shape[0]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    cut = int(n * (1.0 - args.holdout))
    train_idx, test_idx = order[:cut], order[cut:]
    train = {k: v[train_idx] for k, v in data.items()}
    test = {k: v[test_idx] for k, v in data.items()} if len(test_idx) else train
    w0_mg = np.array(W_MG, np.float64)
    w0_eg = np.array(W_EG, np.float64)
    before = mse(test, w0_mg, w0_eg, args.k)
    w_mg, w_eg, train_mse = adam_tune(
        train,
        epochs=args.epochs,
        lr=args.lr,
        k=args.k,
        l2=args.l2,
        batch=args.batch,
        seed=args.seed,
    )
    after = mse(test, w_mg, w_eg, args.k)
    rounded_mg = np.array([round(x) for x in w_mg], np.float64)
    rounded_eg = np.array([round(x) for x in w_eg], np.float64)
    report = {
        "positions": n,
        "train": int(train_idx.shape[0]),
        "holdout": int(test_idx.shape[0]),
        "k": args.k,
        "mse_start": before,
        "mse_train": train_mse,
        "mse_holdout": after,
        "mse_holdout_rounded": mse(test, rounded_mg, rounded_eg, args.k),
        "w_mg": [round(x) for x in w_mg],
        "w_eg": [round(x) for x in w_eg],
        "terms": list(TERM_NAMES),
    }
    print(json.dumps(report, indent=2), flush=True)
    print(format_weights(w_mg, w_eg), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def _cmd_mse(args: argparse.Namespace) -> int:
    loaded = np.load(args.data)
    keys = ("feat", "pst_mg", "pst_eg", "phase", "result")
    data = {k: loaded[k].astype(np.float64) for k in keys}
    print(
        json.dumps(
            {
                "positions": int(data["result"].shape[0]),
                "mse": mse(data, np.array(W_MG, np.float64), np.array(W_EG, np.float64), args.k),
                "k": args.k,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    ext = sub.add_parser("extract", help="build a quiet-labelled npz from PGNs")
    ext.add_argument("pgn", nargs="+")
    ext.add_argument("-o", "--output", type=Path, required=True)
    ext.add_argument("--min-ply", type=int, default=16)
    ext.add_argument("--max-positions", type=int)
    ext.set_defaults(run=_cmd_extract)

    tun = sub.add_parser("tune", help="Adam-tune extra-term weights")
    tun.add_argument("data", type=Path)
    tun.add_argument("-o", "--output", type=Path)
    tun.add_argument("--epochs", type=int, default=40)
    tun.add_argument("--lr", type=float, default=0.05)
    tun.add_argument("--k", type=float, default=110.0)
    tun.add_argument("--l2", type=float, default=2e-4)
    tun.add_argument("--batch", type=int, default=4096)
    tun.add_argument("--holdout", type=float, default=0.1)
    tun.add_argument("--seed", type=int, default=1)
    tun.set_defaults(run=_cmd_tune)

    m = sub.add_parser("mse", help="MSE of the current baked-in weights")
    m.add_argument("data", type=Path)
    m.add_argument("--k", type=float, default=110.0)
    m.set_defaults(run=_cmd_mse)

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
