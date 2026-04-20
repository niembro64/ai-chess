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

from . import tablebase
from .encoding import POLICY_SIZE, encode_board, move_to_index
from .engine import (
    CastlingRights,
    ChessGameState,
    Move,
    Piece,
    PieceColor,
    PieceType,
    apply_move,
    create_initial_game_state,
    get_legal_moves,
    is_in_check,
)
from .mcts import run_batched_mcts
from .model import NUM_PLANES, ChessNet, encoded_to_nchw
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights


# --- Data ---


@dataclass
class TrainingExample:
    board: np.ndarray       # [8*8*NUM_PLANES]
    policy: np.ndarray      # [POLICY_SIZE]
    value: float            # [-1, 1]
    # True iff we actually know the outcome: mate, stalemate, 50-move,
    # or any tb-adjudicated result. False only for cap-timeout games
    # where we ran out of plies with too many pieces for the tablebase
    # to adjudicate — the value=0 label for those is a guess, not a
    # known draw. The trainer masks these out of the value-loss
    # computation so the value head isn't pulled toward "predict draw"
    # by noise. Policy signal (MCTS visit distribution) is still used.
    outcome_known: bool = True


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
    # How the game was initialized. Used to split outcome stats so we can
    # tell whether the endgame curriculum is producing real play signal
    # (mate_w/mate_b) vs mostly tablebase-distilled labels (tb_w/tb_b).
    # One of: "standard" | "endgame" | "random".
    origin: str = "standard"


@dataclass
class GameResult:
    move_count: int
    # Granular end-state label. The trainer's _record_outcome dispatches
    # on this to increment the right bucket in TrainStats:
    #   mate_w    — over-the-board checkmate, white won
    #   mate_b    — over-the-board checkmate, black won
    #   stalemate — no legal moves, not in check
    #   draw_50   — 50-move rule
    #   tb_w      — cap-timeout, Syzygy adjudicated as white win
    #   tb_b      — cap-timeout, Syzygy adjudicated as black win
    #   tb_d      — cap-timeout, Syzygy adjudicated as draw
    #   cap       — cap-timeout, no tablebase signal (scored 0)
    outcome: str
    outcome_label: str        # human string for logging
    white_outcome: float      # [-1, 1]
    tb_adjudicated: bool = False  # True iff outcome came from Syzygy
    origin: str = "standard"  # propagated from GameSlot; see above


# --- Replay buffer (ring, fixed capacity) ---


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._boards = np.zeros((capacity, 8 * 8 * NUM_PLANES), dtype=np.float32)
        self._policies = np.zeros((capacity, POLICY_SIZE), dtype=np.float32)
        self._values = np.zeros((capacity,), dtype=np.float32)
        # Parallel mask: 1.0 when the value label reflects an actual
        # outcome we observed, 0.0 when it's a stand-in for a cap-
        # timeout we couldn't adjudicate. Stored as float32 for direct
        # multiplication into the weighted value loss.
        self._outcome_known = np.ones((capacity,), dtype=np.float32)
        self._size = 0
        self._head = 0

    def __len__(self) -> int:
        return self._size

    def add(self, ex: TrainingExample) -> None:
        i = self._head
        self._boards[i] = ex.board
        self._policies[i] = ex.policy
        self._values[i] = ex.value
        self._outcome_known[i] = 1.0 if ex.outcome_known else 0.0
        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(
        self, batch_size: int, rng: random.Random | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (boards, policies, values, outcome_known_mask).

        The mask is 1.0 for samples where the value label is real and
        0.0 for cap-timeout samples where the value was a no-signal
        guess. Callers should multiply the per-sample value loss by
        this mask to ignore unreliable labels.
        """
        rng = rng or random
        if self._size == 0:
            raise ValueError("Replay buffer is empty")
        idx = np.fromiter((rng.randrange(self._size) for _ in range(batch_size)), dtype=np.int64)
        return (
            self._boards[idx],
            self._policies[idx],
            self._values[idx],
            self._outcome_known[idx],
        )


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


# Endgame material configs for the curriculum starts. Each tuple is
# (strong_side_non_king_pieces, weak_side_non_king_pieces). Kings are added
# automatically. Chosen to be familiar theoretical endgames that resolve
# quickly enough to give the net dense terminal outcomes early in training.
_ENDGAME_CONFIGS: tuple[tuple[tuple[PieceType, ...], tuple[PieceType, ...]], ...] = (
    (("queen",), ()),               # KQvK  — textbook mate in ~10
    (("rook",), ()),                # KRvK  — mate in ~15
    (("rook", "rook"), ()),         # KRRvK — easier than KRvK
    (("queen",), ("rook",)),        # KQvKR — queen usually wins
    (("pawn",), ()),                # KPvK  — teaches promotion + king opposition
    (("pawn", "pawn"), ()),         # KPPvK — multiple promotion paths
    (("queen", "rook"), ()),        # KQRvK — crushing, quick mate
)


def _endgame_start(rng: random.Random) -> ChessGameState:
    """Random simple-endgame starting position drawn from `_ENDGAME_CONFIGS`.

    Retries until we produce a legal, non-terminal position (kings not
    adjacent, opposing king not already in check, side-to-move has a legal
    move). Falls back to the standard opening if we can't after several
    attempts (which in practice never happens on an 8×8 board with ≤5
    pieces).
    """
    strong_pieces, weak_pieces = rng.choice(_ENDGAME_CONFIGS)
    strong_color: PieceColor = "white" if rng.random() < 0.5 else "black"
    weak_color: PieceColor = "black" if strong_color == "white" else "white"

    for _ in range(30):
        board: list[list[Piece | None]] = [[None] * 8 for _ in range(8)]
        squares = [(r, f) for r in range(8) for f in range(8)]
        rng.shuffle(squares)
        sq_iter = iter(squares)

        try:
            sk_r, sk_f = next(sq_iter)
            wk_r, wk_f = next(sq_iter)
        except StopIteration:
            continue
        # Kings can't be adjacent (illegal position).
        if abs(sk_r - wk_r) <= 1 and abs(sk_f - wk_f) <= 1:
            continue
        board[sk_r][sk_f] = Piece(strong_color, "king")
        board[wk_r][wk_f] = Piece(weak_color, "king")

        placed_ok = True
        for color, pieces in ((strong_color, strong_pieces), (weak_color, weak_pieces)):
            for pt in pieces:
                try:
                    r, f = next(sq_iter)
                except StopIteration:
                    placed_ok = False
                    break
                # Pawns can't sit on the promotion ranks.
                if pt == "pawn" and (r == 0 or r == 7):
                    placed_ok = False
                    break
                board[r][f] = Piece(color, pt)
            if not placed_ok:
                break
        if not placed_ok:
            continue

        side_to_move: PieceColor = "white" if rng.random() < 0.5 else "black"
        state = ChessGameState(
            board=board,
            currentTurn=side_to_move,
            castlingRights=CastlingRights(False, False, False, False),
            enPassantTarget=None,
            halfMoveClock=0,
            fullMoveNumber=1,
            status="active",
        )
        # Reject illegal setups (opponent in check means they'd have had to
        # move — the position is unreachable) and already-terminal positions.
        opp: PieceColor = "black" if side_to_move == "white" else "white"
        if is_in_check(state.board, opp):
            continue
        if not get_legal_moves(state):
            continue
        return state

    return _normal_start()


def _make_game_slot(
    rng: random.Random,
    random_start_prob: float = 0.3,
    endgame_start_prob: float = 0.0,
) -> GameSlot:
    """Create a fresh game slot — single roll over three buckets.

    The three init types and their probabilities, each sampled from ONE
    uniform roll so the fractions sum to 1.0 literally:

      * `endgame_start_prob` → simple-endgame curriculum position (KQvK,
        KRvK, KPvK, ...) with a medium-short cap (40-120 plies). Dense
        terminal outcomes for early training.
      * `random_start_prob`  → randomized mid-/end-game position via a
        random walk (coverage of uncommon states), short cap (5-30 plies).
      * the remainder        → standard opening, long cap (200-400 plies)
        so real terminal outcomes dominate the value signal.

    If `endgame_start_prob + random_start_prob > 1.0` the standard bucket
    is simply empty (the split is clamped by the roll's upper bound, not
    by renormalization).
    """
    roll = rng.random()
    if roll < endgame_start_prob:
        state = _endgame_start(rng)
        # Raised from (40, 120): KPvK, KRvK and KQvKR routinely need more
        # than 120 plies with a weak model. Too-short caps were driving
        # almost all endgame inits into tablebase adjudication and
        # starving the value head of "real mate" signal.
        move_cap = rng.randint(80, 200)
        return GameSlot(state=state, move_cap=move_cap, is_standard_start=False, origin="endgame")
    if roll < endgame_start_prob + random_start_prob:
        state = _random_start(rng)
        move_cap = rng.randint(5, 30)
        return GameSlot(state=state, move_cap=move_cap, is_standard_start=False, origin="random")
    state = _normal_start()
    # Reverted to (200, 400) after the "longer caps" experiment backfired:
    # lengthening to (400, 600) dropped the `cap` bucket but flooded the
    # `50-move` bucket (from 12% → 54%) and collapsed the value head. With
    # tb-on-50-move now in place the 50-move drain is recovered via the
    # tablebase anyway, so shorter caps are the pragmatic win — more games
    # per wall-clock minute, same decisive-signal yield.
    move_cap = rng.randint(200, 400)
    return GameSlot(state=state, move_cap=move_cap, is_standard_start=True, origin="standard")


# --- Left-right (file) mirror augmentation ---
#
# Chess is symmetric about the file axis, so (board, policy) pairs can be
# flipped for free. Applied at batch-sample time to avoid storing duplicates.

_MIRROR_POLICY_PERM: np.ndarray | None = None


def _compute_mirror_policy_perm() -> np.ndarray:
    """Precompute the policy-index permutation that maps each move (fr,ff)->(tr,tf)
    to its file-mirrored counterpart (fr,7-ff)->(tr,7-tf).

    Satisfies: mirrored_policy[i] = original_policy[perm[i]].
    Because file-mirror is self-inverse, applying `perm` twice yields identity.
    """
    perm = np.zeros(POLICY_SIZE, dtype=np.int64)
    for fr in range(8):
        for ff in range(8):
            for tr in range(8):
                for tf in range(8):
                    src = (fr * 8 + ff) * 64 + (tr * 8 + tf)
                    dst = (fr * 8 + (7 - ff)) * 64 + (tr * 8 + (7 - tf))
                    perm[src] = dst
    return perm


def _get_mirror_policy_perm() -> np.ndarray:
    global _MIRROR_POLICY_PERM
    if _MIRROR_POLICY_PERM is None:
        _MIRROR_POLICY_PERM = _compute_mirror_policy_perm()
    return _MIRROR_POLICY_PERM


def mirror_batch(
    boards: np.ndarray,
    policies: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (boards, policies) with file-mirror applied to the rows where
    `mask` is True. Values are unchanged — they're outcome-relative and
    don't depend on board orientation.

    boards:   [B, 8*8*NUM_PLANES]
    policies: [B, POLICY_SIZE]
    mask:     [B] bool
    """
    if not mask.any():
        return boards, policies
    idx = np.where(mask)[0]

    # Board mirror: flip file axis, then swap the castling-rights plane pairs
    # (own K-side <-> own Q-side, opp K-side <-> opp Q-side). Piece/EP planes
    # are spatial so the file flip does the right thing; constant planes
    # (bias/halfmove/fullmove) are unaffected.
    b = boards[idx].reshape(-1, 8, 8, NUM_PLANES)
    b = b[:, :, ::-1, :].copy()
    b[:, :, :, [15, 16, 17, 18]] = b[:, :, :, [16, 15, 18, 17]]
    boards = boards.copy()
    boards[idx] = b.reshape(len(idx), -1)

    # Policy mirror via precomputed permutation.
    perm = _get_mirror_policy_perm()
    policies = policies.copy()
    policies[idx] = policies[idx][:, perm]

    return boards, policies


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


def make_local_selfplay_engine(
    model: ChessNet,
    device: torch.device,
    replay_buffer: ReplayBuffer,
    config: SelfPlayConfig | None = None,
    rng: random.Random | None = None,
) -> SelfPlayEngine:
    """Single-process convenience: build an engine that evaluates with the given
    model on the given device and drops completed examples into `replay_buffer`.
    """
    evaluator = make_pytorch_evaluator(model, device)
    return SelfPlayEngine(evaluator, replay_buffer.add, config, rng)


@dataclass
class SelfPlayConfig:
    num_concurrent_games: int = 32
    mcts_simulations: int = 25
    # Probability that a fresh game slot is seeded from a simple theoretical
    # endgame (KQvK, KRvK, KPvK, ...). Curriculum learning: these resolve
    # in few plies and produce dense terminal outcomes early in training,
    # before the network is strong enough to force mates from the opening.
    endgame_start_prob: float = 0.0
    # Probability that a fresh game slot starts from a randomized mid-/end-game
    # position (with a short move cap). The rest start from the standard
    # opening with a long cap. Higher = more coverage, lower = more tactical
    # depth. 0.3 means 30% random starts, 70% standard.
    random_start_prob: float = 0.3
    # Plies from the start of an episode during which move selection samples
    # proportional to visit counts (τ=1). After this many plies we switch to
    # argmax (τ→0) so the game commits to the best move. Matches the
    # AlphaZero move-selection schedule.
    temperature_threshold_plies: int = 15
    rewards: RewardWeights = field(default_factory=lambda: RewardWeights(**DEFAULT_REWARD_WEIGHTS.__dict__))


# A "sink" accepts completed training examples one at a time. The single-process
# path plugs in `ReplayBuffer.add`; the multiprocessing worker plugs in a
# queue.put so examples flow out of the worker process to the trainer.
ExampleSink = Callable[["TrainingExample"], None]


class SelfPlayEngine:
    def __init__(
        self,
        evaluator: PytorchEvaluator,
        example_sink: ExampleSink,
        config: SelfPlayConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.evaluator = evaluator
        self.example_sink = example_sink
        self.config = config or SelfPlayConfig()
        self.rng = rng or random.Random()

        self.games: list[GameSlot] = [
            _make_game_slot(self.rng, self.config.random_start_prob, self.config.endgame_start_prob)
            for _ in range(self.config.num_concurrent_games)
        ]
        self.games_completed = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.caps = 0
        # Count of games where Syzygy adjudicated a cap into a real outcome.
        # Reported separately from W/B/D so you can see how much lift the
        # tablebase is giving. Included as a tag in GameResult.
        self.tb_adjudications = 0
        self.recent_game_lengths: list[int] = []

    def step(self) -> list[GameResult]:
        """Advance every active game by one move. Returns results for games that ended this step."""
        states = [g.state for g in self.games]
        temperatures = [
            1.0 if g.move_count < self.config.temperature_threshold_plies else 0.0
            for g in self.games
        ]
        mcts_results = run_batched_mcts(
            states, self.evaluator, self.config.mcts_simulations, self.rng, temperatures
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

            # Phase 1 of the reward refactor: pure-outcome value target, so we
            # no longer compute the hand-crafted positional score here.
            # `position_score` is kept on the dataclass for now (set to 0) so
            # external tests and fixtures that build examples directly keep
            # type-compatible. Will be repurposed when aux heads land.
            slot.examples.append(
                GameSlotExample(
                    board=board,
                    policy=canon_policy,
                    turn_color=slot.state.currentTurn,
                    position_score=0.0,
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
                self.games[i] = _make_game_slot(self.rng, self.config.random_start_prob, self.config.endgame_start_prob)

        return finished

    def _finish_game(self, slot: GameSlot) -> GameResult:
        status = slot.state.status
        # `hit_cap` means we terminated via our move_cap, not via a natural
        # terminal state. Distinct from `status == "draw"` which is the
        # engine's 50-move-rule signal.
        hit_cap = slot.move_count >= slot.move_cap and status not in (
            "checkmate",
            "stalemate",
            "draw",
        )
        tb_adjudicated = False

        if status == "checkmate":
            # Winner = player who just moved = opposite of currentTurn.
            winner = "white" if slot.state.currentTurn == "black" else "black"
            white_outcome = 1.0 if winner == "white" else -1.0
            if winner == "white":
                outcome = "mate_w"
                label = "white mates"
                self.white_wins += 1
            else:
                outcome = "mate_b"
                label = "black mates"
                self.black_wins += 1
        else:
            # Non-mate ending — stalemate, 50-move rule, or our own cap.
            # Consult the tablebase on cap-timeouts AND 50-move draws:
            # both are cases where the game drifted to a non-decisive
            # result despite the position being within tb range. A 50-move
            # draw in KQvK is a theoretical win for the stronger side, and
            # overriding the value=0 label with the tb result teaches the
            # value head "don't settle for shuffling" rather than "draws
            # are fine here." Stalemate is a forced legal draw (the side
            # to move genuinely has no moves) and is left alone.
            want_tb_probe = hit_cap or status == "draw"
            tb_result = (
                tablebase.probe_outcome(slot.state) if want_tb_probe else None
            )

            if tb_result is not None:
                stm = slot.state.currentTurn
                # tb_result is from side-to-move's perspective; convert to
                # white's perspective for the outcome convention.
                white_outcome = float(tb_result if stm == "white" else -tb_result)
                source = "50-move" if status == "draw" else "cap"
                if white_outcome > 0:
                    outcome = "tb_w"
                    label = f"{source} → W (tb)"
                    self.white_wins += 1
                elif white_outcome < 0:
                    outcome = "tb_b"
                    label = f"{source} → B (tb)"
                    self.black_wins += 1
                else:
                    outcome = "tb_d"
                    label = f"{source} → D (tb)"
                    self.draws += 1
                self.tb_adjudications += 1
                tb_adjudicated = True
            else:
                # No tablebase signal (disabled / out of range / stalemate).
                white_outcome = 0.0
                if status == "stalemate":
                    outcome = "stalemate"
                    label = "stalemate"
                    self.draws += 1
                elif status == "draw":
                    outcome = "draw_50"
                    label = "50-move"
                    self.draws += 1
                else:
                    # hit_cap with no tb signal.
                    outcome = "cap"
                    label = "cap"
                    self.caps += 1

        win_weight = self.config.rewards.winning
        # Cap-timeouts with no tablebase signal are the only case where
        # value=0 is a guess, not an observation. Every other outcome
        # (mate, stalemate, 50-move, tb_*) has a real label; flag it as
        # such so the trainer weights its value loss accordingly.
        outcome_known = outcome != "cap"

        for ex in slot.examples:
            outcome_from_persp = white_outcome if ex.turn_color == "white" else -white_outcome
            value = outcome_from_persp * win_weight
            value = max(-1.0, min(1.0, value))
            self.example_sink(TrainingExample(
                board=ex.board, policy=ex.policy, value=value,
                outcome_known=outcome_known,
            ))

        self.games_completed += 1
        self.recent_game_lengths.append(slot.move_count)
        if len(self.recent_game_lengths) > 100:
            self.recent_game_lengths = self.recent_game_lengths[-100:]

        return GameResult(
            move_count=slot.move_count,
            outcome=outcome,
            outcome_label=label,
            white_outcome=white_outcome,
            tb_adjudicated=tb_adjudicated,
            origin=slot.origin,
        )
