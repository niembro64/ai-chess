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
from .encoding import POLICY_SIZE, encode_board
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
from .mcts import BatchedEvaluator, run_batched_mcts
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
    # Per-sample weight applied to the policy loss. 1.0 for normal
    # self-play samples; lower for samples from games where the MCTS
    # visit distribution is less trustworthy. Currently used to
    # down-weight TB-adjudicated games (MCTS was meandering — Syzygy
    # rescued the value label but the move choices during play reflect
    # "best guess while lost," not a found forcing line). Value loss
    # weighting is handled separately via outcome_known.
    policy_weight: float = 1.0


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
    # Position-occurrence counter for FIDE threefold-repetition detection.
    # Keyed by `_position_key` (board + side-to-move + castling + ep);
    # halfmove clock excluded since FIDE compares positions modulo it.
    position_history: dict[bytes, int] = field(default_factory=dict)
    # Set by self-play's per-move post-check when it detects a draw the
    # engine doesn't natively report. None = no early termination, fall
    # back to engine status / move-cap. Values: "repetition" |
    # "insufficient_material". The engine doesn't track these because
    # repetition needs game history (engine is stateless across moves)
    # and insufficient-material is FIDE rules glue layered on top of
    # legal-move generation.
    early_termination: str | None = None
    # Set when the side-to-move resigns (MCTS best-child Q <= threshold).
    # The named color loses; opponent gets the win. None = no resign.
    resigned_color: str | None = None
    # Jester mode: which color the LEARNING agent plays in this game.
    # None = both sides are the agent (normal self-play always; in
    # jester mode a None slot is a mirror jester-vs-jester game).
    agent_color: str | None = None
    # Mirror games only: the color playing as the SPARRING PARTNER. Same
    # net and same search as the agent, but its move is sampled at a
    # sustained temperature instead of annealing to greedy, so it
    # blunders. Two greedy loss-seekers never finish a game (neither
    # will deliver the mate the other wants) — the sparring side is what
    # makes the mirror decisive, and it doubles as a model of the human
    # opponent, who is also trying to lose and also imperfect.
    spar_color: str | None = None
    # First color that hit the resign threshold while sampled into the
    # truth-check holdout (resign_disabled_prob). The game plays on;
    # _finish_game compares the predicted loss against the actual
    # outcome to measure the resign false-positive rate.
    resign_truth_color: str | None = None


# position_key moved to engine.py so the MCTS (which this module imports)
# can use it for in-tree repetition awareness without an import cycle.
# Re-exported under the old name for existing callers (train.py, tests).
from .engine import position_key as _position_key


def _is_insufficient_material(board) -> bool:
    """FIDE-strict insufficient material check. Returns True when no
    sequence of legal moves can produce checkmate (per FIDE 5.2.2):
      - K vs K
      - K + bishop vs K
      - K + knight vs K
      - K + bishop vs K + bishop (bishops on same square color)
    NOT included (technically still allow mate via opponent's blunders):
      - K + N vs K + N      → returned as not-insufficient
      - K + B vs K + N      → not-insufficient
      - K + N + N vs K      → not-insufficient (can't *force* but allowed)
    """
    minor_pieces = []  # list of (color, type, square_color)
    for r in range(8):
        for f in range(8):
            p = board[r][f]
            if p is None:
                continue
            if p.type == "king":
                continue
            # Any pawn, rook, or queen → mate is reachable.
            if p.type in ("pawn", "rook", "queen"):
                return False
            minor_pieces.append((p.color, p.type, (r + f) % 2))
    if not minor_pieces:
        return True  # K vs K
    if len(minor_pieces) == 1:
        # Single bishop or knight (against bare king) — insufficient.
        return True
    if len(minor_pieces) == 2:
        a, b = minor_pieces
        if a[1] == "bishop" and b[1] == "bishop":
            # Two bishops, one each side, on same square color → insufficient.
            if a[0] != b[0] and a[2] == b[2]:
                return True
    return False


@dataclass
class GameResult:
    move_count: int
    # Granular end-state label. The trainer's _record_outcome dispatches
    # on this to increment the right bucket in TrainStats:
    #   mate_w            — over-the-board checkmate, white won
    #   mate_b            — over-the-board checkmate, black won
    #   stalemate         — no legal moves, not in check
    #   draw_50           — 50-move rule
    #   draw_repetition   — FIDE threefold repetition
    #   draw_insufficient — FIDE insufficient material (K vs K, etc.)
    #   tb_w              — cap-timeout, Syzygy adjudicated as white win
    #   tb_b              — cap-timeout, Syzygy adjudicated as black win
    #   tb_d              — cap-timeout, Syzygy adjudicated as draw
    #   cap               — cap-timeout, no tablebase signal (scored 0)
    #   resign_w          — black resigned, white wins
    #   resign_b          — white resigned, black wins
    # Resigns get their own buckets (value labels are identical to a
    # mate) so "natural mate rate" diagnostics aren't inflated by games
    # the value head merely predicted as lost.
    outcome: str
    outcome_label: str        # human string for logging
    white_outcome: float      # [-1, 1]
    tb_adjudicated: bool = False  # True iff outcome came from Syzygy
    origin: str = "standard"  # propagated from GameSlot; see above
    # Resign truth-check verdict. None = this game had no held-out
    # resign trigger. False = the would-resign side did lose (threshold
    # was right). True = it did NOT lose (draw or win) — a resign
    # false-positive that would have thrown away a half point or more.
    # Only set when the final outcome is actually known (not cap-guess).
    resign_truth_fp: bool | None = None


# --- Replay buffer (ring, fixed capacity) ---


class ReplayBuffer:
    def __init__(self, capacity: int, num_planes: int = NUM_PLANES):
        self.capacity = capacity
        self._boards = np.zeros((capacity, 8 * 8 * num_planes), dtype=np.float32)
        self._policies = np.zeros((capacity, POLICY_SIZE), dtype=np.float32)
        self._values = np.zeros((capacity,), dtype=np.float32)
        # Parallel mask: 1.0 when the value label reflects an actual
        # outcome we observed, 0.0 when it's a stand-in for a cap-
        # timeout we couldn't adjudicate. Stored as float32 for direct
        # multiplication into the weighted value loss.
        self._outcome_known = np.ones((capacity,), dtype=np.float32)
        # Per-sample weight for the policy loss. 1.0 for regular samples,
        # lower for samples where the MCTS visit distribution is known
        # to be less informative (TB-adjudicated games).
        self._policy_weights = np.ones((capacity,), dtype=np.float32)
        self._size = 0
        self._head = 0
        # Monotonic count of examples ever added. Unlike len(), this keeps
        # growing after the ring fills — gradient-step pacing must diff
        # this, not len(): diffing len() stops counting (and therefore
        # stops ALL gradient steps) the moment the buffer reaches capacity.
        self.total_added = 0

    def __len__(self) -> int:
        return self._size

    def add(self, ex: TrainingExample) -> None:
        self.total_added += 1
        i = self._head
        self._boards[i] = ex.board
        self._policies[i] = ex.policy
        self._values[i] = ex.value
        self._outcome_known[i] = 1.0 if ex.outcome_known else 0.0
        self._policy_weights[i] = float(ex.policy_weight)
        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(
        self, batch_size: int, rng: random.Random | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (boards, policies, values, outcome_known_mask, policy_weights).

        outcome_known_mask is 1.0 for samples where the value label is
        real and 0.0 for cap-timeout samples where the value was a
        no-signal guess. Callers should multiply the per-sample value
        loss by this mask to ignore unreliable labels.

        policy_weights is a per-sample float in [0, 1] — 1.0 for regular
        samples, lower (e.g. 0.5) for TB-adjudicated games where the
        MCTS visit distribution is noisier than usual. Callers should
        multiply the per-sample policy loss by this weight.
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
            self._policy_weights[idx],
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
    num_planes: int = NUM_PLANES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (boards, policies) with file-mirror applied to the rows where
    `mask` is True. Values are unchanged — they're outcome-relative and
    don't depend on board orientation.

    Valid for any encoding that 180°-rotates for black and has no
    file-asymmetric planes needing special handling (Sage's castling
    planes stay UNSWAPPED per the proof in test_mirror_augment.py;
    Toy's 6 piece planes are purely spatial).

    boards:   [B, 8*8*num_planes]
    policies: [B, POLICY_SIZE]
    mask:     [B] bool
    """
    if not mask.any():
        return boards, policies
    idx = np.where(mask)[0]

    # Board mirror: flip the file axis. Castling planes stay UNSWAPPED:
    # because the encoder 180°-rotates for black (rank AND file), white's
    # encoded own-kingside geometry is already the file-mirror of black's
    # encoded own-kingside geometry — so in encoded space the file flip
    # maps own-K-side to own-K-side. (Swapping the plane pairs here, as an
    # earlier version did, decorrelates the castling planes from castle
    # geometry and makes them unlearnable on 50% of samples.) Piece/EP
    # planes are spatial so the file flip does the right thing; constant
    # planes (bias/halfmove/fullmove) are unaffected.
    b = boards[idx].reshape(-1, 8, 8, num_planes)
    b = b[:, :, ::-1, :].copy()
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
            x = encoded_to_nchw(x_flat, getattr(model, "num_planes", NUM_PLANES))
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
    # Per-ply value-target decay. value_i = outcome * decay^plies_from_end.
    # 1.0 = AlphaZero convention (all positions in a winning game label
    # to +1). <1.0 teaches the value head to distinguish mate-in-1 from
    # mate-in-100; essential when MCTS sim count is too low for terminal
    # backup to reliably carry mate-speed info through Q alone.
    value_ply_decay: float = 1.0
    # Soften NN policy priors before feeding them to MCTS during self-play.
    # 1.0 = no change; >1.0 flattens the prior (p**(1/T) + renormalize).
    # Fixes policy-collapse where the trained policy peaks at prior~0.7+
    # on a wrong move — at 100 sims + c_puct=1.5, PUCT can't overcome
    # that peak to visit moves with prior~0.001, so MCTS stays stuck on
    # the bad move and trains the policy to be even more confident.
    # Training target (visit distribution) is unchanged; eval callers
    # leave this at 1.0 so the sharp trained policy is used verbatim.
    policy_softening_temperature: float = 1.0
    # Board encoder for the NN inputs. None = Sage's 20-plane
    # encode_board (and the Rust MCTS fast path stays eligible). Toy
    # passes chess_ai.toy.encode_toy — any non-default encoder forces
    # the Python MCTS path, whose boards are encoded per this callable.
    board_encoder: "Callable[[ChessGameState], np.ndarray] | None" = None
    # --- Jester (misère) mode -------------------------------------------
    # The agent learns to LOSE. Value labels stay truthful; only MCTS
    # selection inverts at agent plies (see mcts.MCTSSearch). Games mix
    # agent-vs-frozen-opponent (dense loss signal — the opponent
    # punishes every offer) with agent-vs-agent mirror games where BOTH
    # sides invert — there a loss must be FORCED against an opponent who
    # refuses to win, which is the strongest form of losing well.
    invert_agent_selection: bool = False
    # Frozen winner net (e.g. the Sage champion) — evaluates its own
    # plies both as game opponent and inside the agent's search tree
    # (dual-net routing). Required for non-mirror jester games.
    frozen_evaluator: "BatchedEvaluator | None" = None
    # Share of fresh slots that are agent-vs-agent mirror games.
    #
    # The mirror is the PRODUCT matchup: both sides want their own king
    # mated, so a loss has to be forced against an opponent who refuses
    # to cooperate (a *selfmate* in chess-problem terms). Playing the
    # frozen winner instead is a *helpmate* — the opponent wants to mate
    # you, so "stop defending" solves it — which is a different and much
    # easier game than the one the bot actually ships into. Keep a small
    # non-mirror share anyway: early on it is the only dense source of
    # "this is what being mated looks like", and it keeps the net honest
    # against a human who accidentally plays a strong winning move.
    agent_selfplay_prob: float = 0.25
    # Move-selection temperature for the sparring side of a mirror game,
    # applied for the WHOLE game (no annealing). >1 is flatter than
    # visit-proportional. Only the played move is affected — the training
    # target is the raw visit distribution either way, so the sparring
    # side's plies stay honest examples.
    spar_temperature: float = 1.0
    # Probability that a sparring ply plays a UNIFORM RANDOM legal move
    # instead of the search's choice. This — not temperature — is what
    # makes mirror games finish: see the long note at the call site in
    # step(). A loss-seeking search never puts visits on its own mating
    # moves, so only an out-of-distribution pick can deliver the mate
    # the opponent wants, which is precisely the accident a human trying
    # to lose commits.
    spar_random_prob: float = 0.0
    # Probability that a sparring ply ACCEPTS a checkmate that is on
    # offer. This is the main source of terminal signal: it turns "the
    # agent manufactured a mating chance" directly into a finished game,
    # which is the skill the whole run is trying to teach. Uniform
    # blunders alone only reached 29.5% checkmate, because landing on a
    # mate by chance already presupposes the skill being learned.
    # Shapes the opponent, not the reward — the mate is real and every
    # value label stays truthful.
    spar_accept_mate_prob: float = 0.0
    # Per-sample policy-loss weight applied to training examples from
    # TB-adjudicated games. In those games MCTS never found a forcing
    # line — Syzygy rescued the value label, but the move distribution
    # at each ply reflects "best guess while lost," not "this is the
    # right plan." Down-weighting the policy signal from those samples
    # stops the policy head from imitating meandering play. 1.0 = no
    # discount; 0.5 halves their influence. Does not affect value loss
    # (that's correct regardless of how it was discovered).
    tb_policy_weight: float = 1.0
    # Resignation: end games early when the side-to-move's MCTS root_value
    # falls at or below this threshold (typical: -0.85 ≈ "near-certain
    # loss"). Saves the cost of running already-decided games to mate or
    # the move cap. The resigning side is recorded as losing.
    # `resign_disabled_prob` skips the resign decision that fraction of the
    # time so we have a truth-check sample of would-be resignations that
    # play on (used to verify the threshold isn't false-positiving).
    # `resign_min_plies` skips the noisy opening where value head is
    # least reliable. Set threshold <= -1.0 to disable resign entirely.
    resign_threshold: float = -0.85
    resign_disabled_prob: float = 0.10
    resign_min_plies: int = 20


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
            self._new_slot() for _ in range(self.config.num_concurrent_games)
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
        # Resignation counters: total plies where threshold was hit and
        # the side actually resigned (resigns), and total plies where
        # threshold was hit but the truth-check sample held the resign
        # back (resign_truth_checks). Useful for tuning the threshold.
        self.resigns: int = 0
        self.resign_truth_checks: int = 0
        # Truth-check verdicts, counted per GAME with a known outcome:
        # finished = held-out games scored, fps = the would-resign side
        # did NOT lose (false positive — resign would have thrown away
        # a half point or more). fps/finished is the resign FP rate the
        # threshold must be tuned against; AZ kept it under ~5%.
        self.resign_truth_finished: int = 0
        self.resign_truth_fps: int = 0

    def _new_slot(self) -> GameSlot:
        slot = _make_game_slot(
            self.rng, self.config.random_start_prob, self.config.endgame_start_prob
        )
        if self.config.invert_agent_selection:
            mirror = (
                self.config.frozen_evaluator is None
                or self.rng.random() < self.config.agent_selfplay_prob
            )
            if mirror:
                # Mirror game: one side spars (sustained temperature) so
                # the game can actually end. Which side is random, so the
                # agent learns to force the loss from both seats.
                slot.spar_color = "white" if self.rng.random() < 0.5 else "black"
            else:
                slot.agent_color = "white" if self.rng.random() < 0.5 else "black"
        return slot

    def step(self) -> list[GameResult]:
        """Advance every active game by one move. Returns results for games that ended this step."""
        states = [g.state for g in self.games]
        temperatures = [
            self.config.spar_temperature
            if g.spar_color is not None and g.state.currentTurn == g.spar_color
            else (1.0 if g.move_count < self.config.temperature_threshold_plies else 0.0)
            for g in self.games
        ]
        jester = self.config.invert_agent_selection
        invert_turns = None
        agent_colors = None
        opponent_evaluator = None
        if jester:
            # Mirror games invert both plies; agent-vs-frozen games invert
            # only the agent's color — in EVERY search of that game, so the
            # frozen side's tree also models the agent truthfully (as a
            # net trying to lose) via the same spec.
            invert_turns = [
                "both" if g.agent_color is None else g.agent_color for g in self.games
            ]
            if self.config.frozen_evaluator is not None:
                opponent_evaluator = self.config.frozen_evaluator
                agent_colors = [g.agent_color for g in self.games]
        mcts_results = run_batched_mcts(
            states,
            self.evaluator,
            self.config.mcts_simulations,
            self.rng,
            temperatures,
            policy_softening_temperature=self.config.policy_softening_temperature,
            board_encoder=self.config.board_encoder,
            # In-tree repetition awareness: each search sees its game's
            # position history, so shuffling reads as a draw in-tree.
            position_counts=[g.position_history for g in self.games],
            invert_turns=invert_turns,
            opponent_evaluator=opponent_evaluator,
            agent_colors=agent_colors,
        )

        finished: list[GameResult] = []

        for i, slot in enumerate(self.games):
            policy = mcts_results[i].policy
            move = mcts_results[i].move

            # Sparring blunder. The played move is replaced by a UNIFORM
            # random legal one — not a hotter sample of the search.
            #
            # Temperature cannot do this job. A loss-seeking search puts
            # near-zero visits on its own mating moves, because
            # delivering mate is the worst thing it can do, so sampling
            # from the visit distribution at any temperature will
            # essentially never deliver the mate the opponent is angling
            # for. The first attempt at this used sustained temperature
            # and mirror games collapsed to 45% move-cap timeouts and
            # 27% threefold, with only 20% reaching a checkmate.
            #
            # A uniform pick can land on a mating move, which is exactly
            # the blunder a human trying to lose makes: they hand you the
            # mate by accident. The training example for this ply still
            # records the honest search distribution — only the move
            # played is randomised, so the position goes off-policy
            # (useful exploration) while the target stays clean.
            #
            # Uniform blunders alone were not enough: they lifted mirror
            # play from 20% to 29.5% checkmate, still far under the 71.5%
            # the old vs-Sage mix produced. The trap is circular. Landing
            # on a mate by chance needs the agent to have arranged a
            # position where mates are plentiful, and the agent cannot
            # learn to arrange one without decisive games to learn from.
            #
            # So the sparring partner also ACCEPTS an offered mate, some
            # of the time. That converts "the agent manufactured a mating
            # chance" straight into a terminal signal, which is precisely
            # the skill worth reinforcing. It shapes the opponent, never
            # the reward: the agent really was checkmated, so every value
            # label stays truthful. Below 1.0 the agent still has to
            # manufacture chances repeatedly rather than bank on one.
            if (
                slot.spar_color is not None
                and slot.state.currentTurn == slot.spar_color
            ):
                options = get_legal_moves(slot.state)
                mates = (
                    [m for m in options
                     if apply_move(slot.state, m).status == "checkmate"]
                    if self.config.spar_accept_mate_prob > 0.0 and options
                    else []
                )
                if mates and self.rng.random() < self.config.spar_accept_mate_prob:
                    move = self.rng.choice(mates)
                elif options and self.rng.random() < self.config.spar_random_prob:
                    move = self.rng.choice(options)

            # Record training example (pre-move state). MCTS places visit
            # mass only on legal-move indices, so its returned policy IS
            # the canonical training target — an earlier version re-derived
            # the legal moves and copied the same entries index-by-index,
            # a bit-identical no-op costing one get_legal_moves per ply.
            # In jester games vs the frozen opponent, only the AGENT's
            # plies become training examples — the frozen net's moves are
            # environment, not behavior to learn.
            record_example = (
                not jester
                or slot.agent_color is None
                or slot.state.currentTurn == slot.agent_color
            )
            board = (self.config.board_encoder or encode_board)(slot.state)

            # Phase 1 of the reward refactor: pure-outcome value target, so we
            # no longer compute the hand-crafted positional score here.
            # `position_score` is kept on the dataclass for now (set to 0) so
            # external tests and fixtures that build examples directly keep
            # type-compatible. Will be repurposed when aux heads land.
            if record_example:
                slot.examples.append(
                    GameSlotExample(
                        board=board,
                        policy=policy,
                        turn_color=slot.state.currentTurn,
                        position_score=0.0,
                    )
                )

            # Resignation: trigger on the best VISITED child's Q ("even
            # my best move loses"), from the side-to-move's perspective.
            # NOT on root_value — the root's mean Q averages over the
            # forced exploration of the mover's own bad moves, which
            # biases it pessimistic and over-fires resignation. If best_q
            # is at or below the threshold past the noisy opening, this
            # side is essentially lost. End the game now (saves the cost
            # of playing it out to mate / cap), unless we're sampled into
            # the truth-check holdout (resign_disabled_prob fraction).
            # The resigning side loses; the opponent gets the win in
            # `_finish_game`.
            #
            # The training example for this position is already recorded
            # above — its value label will be propagated as a normal loss
            # for the resigning side via the per-ply value-decay logic.
            rt = self.config.resign_threshold
            if (
                rt > -1.0
                and slot.move_count >= self.config.resign_min_plies
                and mcts_results[i].best_q <= rt
            ):
                if self.rng.random() < self.config.resign_disabled_prob:
                    # Truth-check sample — do not resign; play on so
                    # _finish_game can compare predicted-loss to the
                    # actual outcome (false-positive measurement).
                    self.resign_truth_checks += 1
                    if slot.resign_truth_color is None:
                        slot.resign_truth_color = slot.state.currentTurn
                else:
                    self.resigns += 1
                    slot.resigned_color = slot.state.currentTurn
                    finished.append(self._finish_game(slot))
                    self.games[i] = self._new_slot()
                    continue

            # Apply the move.
            slot.state = apply_move(slot.state, move)
            slot.move_count += 1

            # FIDE draw detection that the engine doesn't catch on its own:
            # threefold repetition needs game history (engine is stateless),
            # and insufficient material is rules glue not in legal-move gen.
            # Skip the check when the engine already produced a terminal
            # status (checkmate / stalemate / 50-move draw) — those are
            # already game-ending and dispatching on early_termination
            # would just shadow them.
            if slot.state.status not in ("checkmate", "stalemate", "draw"):
                key = _position_key(slot.state)
                slot.position_history[key] = slot.position_history.get(key, 0) + 1
                if slot.position_history[key] >= 3:
                    slot.early_termination = "repetition"
                elif _is_insufficient_material(slot.state.board):
                    slot.early_termination = "insufficient_material"

            is_over = (
                slot.state.status in ("checkmate", "stalemate", "draw")
                or slot.early_termination is not None
                or slot.move_count >= slot.move_cap
            )
            if is_over:
                finished.append(self._finish_game(slot))
                self.games[i] = self._new_slot()

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

        if slot.resigned_color is not None:
            # Resign branch — short-circuits all the natural-end logic.
            # Resigning side loses; opponent wins. Training value labels
            # are identical to a mate, but the outcome bucket is kept
            # separate (resign_w/resign_b) so mate-rate diagnostics
            # measure games the search actually finished.
            if slot.resigned_color == "white":
                outcome = "resign_b"
                label = "white resigns"
                white_outcome = -1.0
                self.black_wins += 1
            else:
                outcome = "resign_w"
                label = "black resigns"
                white_outcome = 1.0
                self.white_wins += 1
        elif status == "checkmate":
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
        elif slot.early_termination == "repetition":
            # FIDE threefold repetition — same position seen 3+ times.
            # Real, FIDE-recognized draw; not a cap-timeout, no TB probe.
            outcome = "draw_repetition"
            label = "3-fold repetition"
            white_outcome = 0.0
            self.draws += 1
        elif slot.early_termination == "insufficient_material":
            # FIDE insufficient material (K vs K, K+B vs K, etc.). No
            # mate is reachable; outcome is forced-draw and known.
            outcome = "draw_insufficient"
            label = "insufficient material"
            white_outcome = 0.0
            self.draws += 1
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
        # Policy-loss weight: discount games whose MCTS visit
        # distributions came from play that never found a forcing line —
        # TB-adjudicated games (Syzygy labeled the outcome post-hoc) AND
        # unadjudicated cap games (the most meandering play of all; an
        # earlier version discounted only the TB case, which left the
        # least informative distributions at full weight). See config
        # docstring. Applied per-game.
        policy_weight = (
            self.config.tb_policy_weight
            if (tb_adjudicated or not outcome_known)
            else 1.0
        )

        # Per-ply value decay: positions close to terminal get magnitude
        # ≈1, positions further away decay toward 0. Teaches the value
        # head to rank mate-in-1 > mate-in-20 > "merely winning" —
        # otherwise MCTS Q is undifferentiated across all winning moves
        # and can't guide search toward faster wins. The last example
        # in slot.examples is the position the winning move was played
        # from — it gets the undecayed label (decay^0 = ±1).
        #
        # CAUTION if re-enabling decay < 1: the WDL conversion turns
        # decayed magnitude into draw probability (P(draw) = 1 - |v|),
        # so aggressive decay labels early-game positions of decisive
        # games as draws. That corrupted a full training run at 0.99
        # (P(draw) > 0.5 beyond ~69 plies from the end). Keep any decay
        # mild enough that |v| stays near 1 over real game lengths.
        decay = self.config.value_ply_decay
        n = len(slot.examples)
        for i, ex in enumerate(slot.examples):
            plies_from_end = n - 1 - i
            ply_factor = decay ** plies_from_end if decay != 1.0 else 1.0
            outcome_from_persp = white_outcome if ex.turn_color == "white" else -white_outcome
            value = outcome_from_persp * ply_factor * win_weight
            value = max(-1.0, min(1.0, value))
            self.example_sink(TrainingExample(
                board=ex.board, policy=ex.policy, value=value,
                outcome_known=outcome_known,
                policy_weight=policy_weight,
            ))

        self.games_completed += 1
        self.recent_game_lengths.append(slot.move_count)
        if len(self.recent_game_lengths) > 100:
            self.recent_game_lengths = self.recent_game_lengths[-100:]

        # Resign truth-check: this game had a held-out resign trigger
        # and played on. Compare the prediction ("resign_truth_color is
        # lost") against what actually happened. Only score it when the
        # outcome is real — cap-timeouts without adjudication are a
        # guess, not a verdict.
        resign_truth_fp: bool | None = None
        if slot.resign_truth_color is not None and outcome_known:
            lost = (
                white_outcome < 0.0
                if slot.resign_truth_color == "white"
                else white_outcome > 0.0
            )
            resign_truth_fp = not lost
            self.resign_truth_finished += 1
            if resign_truth_fp:
                self.resign_truth_fps += 1

        return GameResult(
            move_count=slot.move_count,
            outcome=outcome,
            outcome_label=label,
            white_outcome=white_outcome,
            tb_adjudicated=tb_adjudicated,
            origin=slot.origin,
            resign_truth_fp=resign_truth_fp,
        )
