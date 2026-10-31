"""A tiny deterministic UCI engine for testing the tooling: takes the most valuable capture."""

from __future__ import annotations

import sys

import chess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def choose(board: chess.Board) -> chess.Move:
    def key(move: chess.Move) -> tuple[int, str]:
        captured = board.piece_type_at(move.to_square)
        if board.is_en_passant(move):
            captured = chess.PAWN
        return (-VALUES.get(captured or 0, 0), move.uci())

    return min(board.legal_moves, key=key)


def main() -> None:
    board = chess.Board()
    for raw in sys.stdin:
        tokens = raw.split()
        if not tokens:
            continue
        command = tokens[0]
        if command == "uci":
            print("id name GreedyFixture\nid author chessbot tests\nuciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            if tokens[1] == "startpos":
                board, rest = chess.Board(), tokens[2:]
            else:
                board, rest = chess.Board(" ".join(tokens[2:8])), tokens[8:]
            for uci in rest[1:] if rest[:1] == ["moves"] else []:
                board.push_uci(uci)
        elif command == "go":
            move = choose(board)
            nodes = board.legal_moves.count()
            print(f"info depth 1 nodes {nodes} score cp 0 pv {move.uci()}", flush=True)
            print(f"bestmove {move.uci()}", flush=True)
        elif command == "quit":
            break


if __name__ == "__main__":
    main()
