"""Self-play engine — port of the game-generation half of `Trainer.ts`.

Runs N concurrent games in lockstep, advancing each by one move per step
using batched MCTS. When a game ends, its accumulated (board, policy, value)
tuples get pushed into a ring-buffer replay buffer ready for gradient updates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .encoding import POLICY_SIZE, encode_board, move_to_index
from .engine import ChessGameState, Move, apply_move, create_initial_game_state, get_legal_moves
from .mcts import run_batched_mcts
from .model import NUM_PLANES, ChessNet, encoded_to_nchw
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights, evaluate_position


# --- Data ---


@dataclass
class TrainingExample:
    board: np.ndarray       # [8*8*NUM_PLANES]
    policy: np.ndarray      # [POLICY_SIZE]
    value: float            # [-1, 1]


@dataclass
class GameSlotExample:
    board: np.ndarray
    policy: np.ndarray
    turn_color: str          # "white" | "black"
    position_score: float    # Shaped reward from turn_color's perspective


@dataclass
class GameSlot:
    state: ChessGameState
    examples: list[GameSlotExample] = field(default_factory=list)
    move_count: int = 0
    move_cap: int = 100
    is_standard_start: bool = False


@dataclass
class GameResult:
    move_count: int
    outcome: str              # "white", "black", "draw", "cap"
    outcome_label: str        # human string for logging
    white_outcome: float      # [-1, 1]


# --- Replay buffer (ring, fixed capacity) ---


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._boards = np.zeros((capacity, 8 * 8 * NUM_PLANES), dtype=np.float32)
        self._policies = np.zeros((capacity, POLICY_SIZE), dtype=np.float32)
        self._values = np.zeros((capacity,), dtype=np.float32)
        self._size = 0
        self._head = 0

    def __len__(self) -> int:
        return self._size

    def add(self, ex: TrainingExample) -> None:
        i = self._head
        self._boards[i] = ex.board
        self._policies[i] = ex.policy
        self._values[i] = ex.value
        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = rng or random
        if self._size == 0:
            raise ValueError("Replay buffer is empty")
        idx = np.fromiter((rng.randrange(self._size) for _ in range(batch_size)), dtype=np.int64)
        return self._boards[idx], self._policies[idx], self._values[idx]


# --- Starting position helpers (mirrors Trainer.ts) ---


def _normal_start() -> ChessGameState:
    s = create_initial_game_state()
    s.status = "active"
    return s


def _random_start(rng: random.Random) -> ChessGameState:
    # 40% opening, 30% midgame, 30% late game — matches Trainer.ts.
    roll = rng.random()
    if roll < 0.4:
        num_random = rng.randint(0, 6)
    elif roll < 0.7:
        num_random = rng.randint(12, 24)
    else:
        num_random = rng.randint(30, 50)

    state = _normal_start()
    for _ in range(num_random):
        legal = get_legal_moves(state)
        if not legal:
            break
        state = apply_move(state, rng.choice(legal))
        if state.status in ("checkmate", "stalemate", "draw"):
            return _normal_start()
    return state


def _make_game_slot(rng: random.Random, use_random: bool | None = None) -> GameSlot:
    is_random = use_random if use_random is not None else (rng.random() > 0.1)
    state = _random_start(rng) if is_random else _normal_start()
    base_cap = rng.randint(5, 30)
    move_cap = base_cap if is_random else base_cap * 10
    return GameSlot(state=state, move_cap=move_cap, is_standard_start=not is_random)


# --- Material advantage helper (for capped-game scoring) ---

_PIECE_POINTS = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}


def _material_advantage_white(state: ChessGameState) -> float:
    own = 0
    opp = 0
    for r in range(8):
        for f in range(8):
            piece = state.board[r][f]
            if piece is None:
                continue
            val = _PIECE_POINTS.get(piece.type, 0)
            if piece.color == "white":
                own += val
            else:
                opp += val
    return (own - opp) / 39.0


# --- Self-play engine ---


PytorchEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def make_pytorch_evaluator(model: ChessNet, device: torch.device) -> PytorchEvaluator:
    """Wrap a PyTorch ChessNet as the batched evaluator MCTS expects.

    Returns (policies, values) where values = P(win) - P(loss).
    """
    def evaluate(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x_flat = torch.from_numpy(boards).to(device)
            x = encoded_to_nchw(x_flat, NUM_PLANES)
            policy, wdl = model(x)
        pol = policy.cpu().numpy()
        w = wdl.cpu().numpy()
        values = w[:, 0] - w[:, 2]
        return pol, values

    return evaluate


@dataclass
class SelfPlayConfig:
    num_concurrent_games: int = 32
    mcts_simulations: int = 25
    rewards: RewardWeights = field(default_factory=lambda: RewardWeights(**DEFAULT_REWARD_WEIGHTS.__dict__))


class SelfPlayEngine:
    def __init__(
        self,
        model: ChessNet,
        device: torch.device,
        replay_buffer: ReplayBuffer,
        config: SelfPlayConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.model = model
        self.device = device
        self.replay = replay_buffer
        self.config = config or SelfPlayConfig()
        self.rng = rng or random.Random()

        self.games: list[GameSlot] = [
            _make_game_slot(self.rng, use_random=self.rng.random() > 0.1)
            for _ in range(self.config.num_concurrent_games)
        ]
        self.games_completed = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.recent_game_lengths: list[int] = []

    def step(self) -> list[GameResult]:
        """Advance every active game by one move. Returns results for games that ended this step."""
        evaluator = make_pytorch_evaluator(self.model, self.device)
        states = [g.state for g in self.games]
        mcts_results = run_batched_mcts(
            states, evaluator, self.config.mcts_simulations, self.rng
        )

        finished: list[GameResult] = []

        for i, slot in enumerate(self.games):
            policy = mcts_results[i].policy
            move = mcts_results[i].move

            # Record training example (pre-move state).
            board = encode_board(slot.state)
            legal = get_legal_moves(slot.state)
            is_white = slot.state.currentTurn == "white"
            canon_policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            seen: set[int] = set()
            for m in legal:
                mi = move_to_index(m, is_white)
                if mi not in seen:
                    seen.add(mi)
                    canon_policy[mi] = policy[mi]

            pos_score = evaluate_position(
                slot.state, slot.state.currentTurn, self.config.rewards, len(legal)
            )
            slot.examples.append(
                GameSlotExample(
                    board=board,
                    policy=canon_policy,
                    turn_color=slot.state.currentTurn,
                    position_score=pos_score,
                )
            )

            # Apply the move.
            slot.state = apply_move(slot.state, move)
            slot.move_count += 1

            is_over = (
                slot.state.status in ("checkmate", "stalemate", "draw")
                or slot.move_count >= slot.move_cap
            )
            if is_over:
                finished.append(self._finish_game(slot))
                self.games[i] = _make_game_slot(self.rng)

        return finished

    def _finish_game(self, slot: GameSlot) -> GameResult:
        hit_cap = slot.move_count >= slot.move_cap and slot.state.status not in (
            "checkmate",
            "stalemate",
            "draw",
        )

        if slot.state.status == "checkmate":
            # The CURRENT player (after the winning move was applied) has no moves.
            # winner = player who JUST moved = opposite of currentTurn.
            winner = "white" if slot.state.currentTurn == "black" else "black"
            white_outcome = 1.0 if winner == "white" else -1.0
            outcome = winner
            label = f"{winner} wins"
            if winner == "white":
                self.white_wins += 1
            else:
                self.black_wins += 1
        elif hit_cap:
            mat_adv = _material_advantage_white(slot.state)
            white_outcome = max(-1.0, min(1.0, mat_adv * 3.0))
            outcome = "cap"
            label = f"cap ({mat_adv * 39:+.0f})"
            self.draws += 1
        else:
            white_outcome = 0.0
            outcome = "draw"
            label = "stalemate" if slot.state.status == "stalemate" else "draw"
            self.draws += 1

        win_weight = self.config.rewards.winning

        for ex in slot.examples:
            outcome_from_persp = white_outcome if ex.turn_color == "white" else -white_outcome
            value = outcome_from_persp * win_weight + ex.position_score
            value = max(-1.0, min(1.0, value))
            self.replay.add(TrainingExample(board=ex.board, policy=ex.policy, value=value))

        self.games_completed += 1
        self.recent_game_lengths.append(slot.move_count)
        if len(self.recent_game_lengths) > 100:
            self.recent_game_lengths = self.recent_game_lengths[-100:]

        return GameResult(
            move_count=slot.move_count,
            outcome=outcome,
            outcome_label=label,
            white_outcome=white_outcome,
        )
