from __future__ import annotations

import io

import chess.pgn
import numpy as np

from chessbot.core.board import Board
from chessbot.search.evaluate import N_TERMS, W_EG, W_MG, build_eval_tables, evaluate
from chessbot.tools.texel import adam_tune, extract_game, mse


def test_extract_skips_checks_and_captures() -> None:
    pgn = """
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Nxd5 6. Nxf7 Kxf7 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    feats, _mg, _eg, _phase, labels = extract_game(game, min_ply=0)
    assert labels
    assert all(f.shape == (N_TERMS,) for f in feats)
    # The capture 5.exd5 and the check 6.Nxf7 must not appear as labelled positions.
    board = game.board()
    quiet_fens = []
    for move in game.mainline_moves():
        capture = board.is_capture(move)
        board.push(move)
        if not capture and not board.is_check() and not board.is_game_over():
            quiet_fens.append(board.fen())
    assert len(feats) == len(quiet_fens) == len(labels)


def test_adam_reduces_mse_on_synthetic_features() -> None:
    rng = np.random.default_rng(0)
    n = 800
    feat = rng.normal(size=(n, N_TERMS))
    feat[:, :] = np.round(feat)
    true_mg = np.array(W_MG, np.float64) + rng.normal(0, 4, N_TERMS)
    true_eg = np.array(W_EG, np.float64) + rng.normal(0, 4, N_TERMS)
    phase = rng.integers(0, 25, size=n).astype(np.float64)
    pst_mg = rng.normal(0, 40, size=n)
    pst_eg = rng.normal(0, 40, size=n)
    mg = pst_mg + feat @ true_mg
    eg = pst_eg + feat @ true_eg
    score = (mg * phase + eg * (24 - phase)) / 24
    p = 1.0 / (1.0 + np.exp(-score / 110.0))
    result = (rng.random(n) < p).astype(np.float64)
    data = {
        "feat": feat,
        "pst_mg": pst_mg,
        "pst_eg": pst_eg,
        "phase": phase,
        "result": result,
    }
    before = mse(data, np.array(W_MG, np.float64), np.array(W_EG, np.float64), 110.0)
    _w_mg, _w_eg, after = adam_tune(data, epochs=15, lr=0.1, k=110.0, l2=0.0, batch=200, seed=0)
    assert after < before


def test_eval_tables_accept_custom_weights() -> None:
    board = Board()
    ev = build_eval_tables(np.zeros(N_TERMS, np.int32), np.zeros(N_TERMS, np.int32))
    assert ev.w_mg.shape == (N_TERMS,)
    assert evaluate(board.pos, board.t, ev) == 0
