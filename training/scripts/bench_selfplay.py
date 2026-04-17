"""Micro-benchmark for the self-play pipeline hot paths.

Times the three dominant costs (MCTS expansion, get_legal_moves, encode_board)
plus end-to-end `SelfPlayEngine.step()` latency. Use before/after an
optimization to prove the speedup is real.

    python scripts/bench_selfplay.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from chess_ai.encoding import encode_board
from chess_ai.engine import apply_move, create_initial_game_state, get_legal_moves
from chess_ai.model import ChessNet
from chess_ai.selfplay import ReplayBuffer, SelfPlayConfig, SelfPlayEngine


def bench_get_legal_moves(n: int = 1000) -> float:
    """Play random moves, timing get_legal_moves on each reachable position."""
    rng = random.Random(0)
    state = create_initial_game_state()
    state.status = "active"

    positions = []
    while len(positions) < n:
        legal = get_legal_moves(state)
        if not legal:
            state = create_initial_game_state()
            state.status = "active"
            continue
        positions.append(state)
        state = apply_move(state, rng.choice(legal))
        if state.status in ("checkmate", "stalemate", "draw"):
            state = create_initial_game_state()
            state.status = "active"

    t = time.perf_counter()
    total_moves = 0
    for s in positions:
        total_moves += len(get_legal_moves(s))
    elapsed = time.perf_counter() - t
    print(f"get_legal_moves: {n} calls, {total_moves} moves returned, "
          f"{elapsed:.3f}s total → {elapsed / n * 1000:.3f} ms/call")
    return elapsed


def bench_encode_board(n: int = 5000) -> float:
    """Time encode_board on a mixed set of positions reached by random play."""
    rng = random.Random(1)
    state = create_initial_game_state()
    state.status = "active"

    positions = []
    while len(positions) < n:
        legal = get_legal_moves(state)
        if not legal:
            state = create_initial_game_state()
            state.status = "active"
            continue
        positions.append(state)
        state = apply_move(state, rng.choice(legal))
        if state.status in ("checkmate", "stalemate", "draw"):
            state = create_initial_game_state()
            state.status = "active"

    t = time.perf_counter()
    for s in positions:
        encode_board(s)
    elapsed = time.perf_counter() - t
    print(f"encode_board: {n} calls, {elapsed:.3f}s total → "
          f"{elapsed / n * 1e6:.1f} µs/call")
    return elapsed


def bench_selfplay_step(
    num_steps: int = 5,
    concurrent_games: int = 32,
    mcts_sims: int = 25,
    device: str = "cpu",
) -> float:
    """Time a handful of SelfPlayEngine.step() calls end-to-end."""
    torch.manual_seed(0)
    dev = torch.device(device)
    model = ChessNet(
        num_res_blocks=6, num_filters=64, kernel_size=3,
        value_head_size=32, se_reduction=8,
    ).to(dev)
    model.eval()

    buf = ReplayBuffer(capacity=10_000)
    engine = SelfPlayEngine(
        model=model, device=dev, replay_buffer=buf,
        config=SelfPlayConfig(num_concurrent_games=concurrent_games, mcts_simulations=mcts_sims),
        rng=random.Random(42),
    )

    # Warm-up step (NN JIT / first CUDA dispatch).
    engine.step()

    t = time.perf_counter()
    for _ in range(num_steps):
        engine.step()
    elapsed = time.perf_counter() - t
    per_step = elapsed / num_steps
    print(
        f"selfplay.step ({concurrent_games}g × {mcts_sims}sims, {device}): "
        f"{num_steps} steps in {elapsed:.2f}s → {per_step * 1000:.0f} ms/step"
    )
    return per_step


if __name__ == "__main__":
    print("=== Hot-path micro-benchmarks ===\n")
    bench_get_legal_moves(n=500)
    bench_encode_board(n=3000)
    print()
    print("=== End-to-end self-play ===\n")
    bench_selfplay_step(num_steps=3, concurrent_games=16, mcts_sims=15, device="cpu")
