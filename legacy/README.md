# Legacy bot

The original bot, kept for reference. Nothing in `src/` imports it.

| File | What it is |
|---|---|
| `chess_ai.ipynb` | Notebook that trains a Keras CNN to predict the move a Lichess Elite player made, then plays the highest-probability legal move (no search) |
| `TF_10EPOCHS.keras`, `TF_50EPOCHS.keras` | The trained models. The class-index-to-move mapping was never saved, so they cannot be used reproducibly in a new Python process |
| `legacy-selfplay-game.pgn` | The only game the bot played, copied from the notebook's last output (83 moves, playing both sides) |

## Baseline

The legacy game is the accuracy baseline for the new engine:

```bash
chessbot-acpl legacy/legacy-selfplay-game.pgn --depth 14
```

With Stockfish 16 at depth 14, evals clamped to ±1000 cp, and each move's loss capped at 1000 cp:

| Side | Moves | ACPL | Blunders (≥300 cp) | Mistakes (100–299 cp) |
|---|---|---|---|---|
| White | 83 | 366 | 35 | 14 |
| Black | 82 | 347 | 32 | 13 |
| Both | 165 | 356 | 67 | 27 |

For comparison, club players typically average 50–100 ACPL.
