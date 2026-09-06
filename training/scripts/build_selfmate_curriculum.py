"""Regenerate and independently prove the checked-in short selfmate catalog."""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_ai.inverted import forced_selfmate_moves, ProofBudgetExceeded


def main():
    rng = random.Random(20260906)
    candidates = []
    # Discovered rook mates; vary the blockers, checking piece, and black king.
    for king in ["b8", "c8", "d8", "e8"]:
        for bishop in ["f5", "g6", "e4", "d3"]:
            for knight in ["h5", "f5", "e6"]:
                for queen in chess.SQUARES:
                    b = chess.Board.empty()
                    for sq, symbol in [("h8", "K"), ("a8", "r"), (king, "k"), (bishop, "b"), (knight, "n")]:
                        b.set_piece_at(chess.parse_square(sq), chess.Piece.from_symbol(symbol))
                    if b.piece_at(queen):
                        continue
                    b.set_piece_at(queen, chess.Piece(chess.QUEEN, chess.WHITE))
                    b.turn = chess.WHITE
                    if b.is_valid() and not b.is_check():
                        candidates.append(b.fen())
    rng.shuffle(candidates)
    rows = []
    counts = {2: 0, 4: 0, 6: 0}
    for fen in dict.fromkeys(candidates):
        try:
            for plies in [2, 4, 6]:
                moves = forced_selfmate_moves(fen, plies, node_budget=25_000)
                if moves:
                    if counts[plies] < 12:
                        rows.append(
                            dict(
                                fen=fen,
                                plies=plies,
                                winning_moves=moves,
                                split="eval" if counts[plies] % 4 == 3 else "train",
                            )
                        )
                        counts[plies] += 1
                        print(counts, fen, flush=True)
                    break
        except ProofBudgetExceeded:
            continue
        if all(n >= 12 for n in counts.values()):
            break
    if any(n < 4 for n in counts.values()):
        raise RuntimeError(f"Insufficient proof coverage: {counts}")
    path = Path(__file__).resolve().parents[1] / "src/chess_ai/selfmate_positions.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print("Wrote", path, len(rows))


if __name__ == "__main__":
    main()
