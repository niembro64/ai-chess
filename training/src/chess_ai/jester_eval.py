"""Batched, resistant-opponent evaluation for competitive inverted chess."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .engine import apply_move, get_legal_moves, position_key
from .eval_positions import build_eval_positions, build_rotating_opening_positions
from .inverted import selfmate_positions
from .mcts import run_batched_mcts
from .selfplay import _is_insufficient_material


@dataclass
class Match:
    state: object
    color: str
    opponent: object
    pair: str
    difficulty: str
    plies: int = 0
    history: dict = field(default_factory=dict)


def move_uci(move) -> str:
    def square(p):
        return f"{chr(97 + p.file)}{8 - p.rank}"

    suffix = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}.get(move.promotion, "")
    return square(move.from_pos) + square(move.to_pos) + suffix


def terminal_result(match: Match, cap: int) -> str | None:
    state = match.state
    if state.status == "checkmate":
        return "win" if state.currentTurn == match.color else "loss"
    if state.status in ("draw", "stalemate") or _is_insufficient_material(state.board):
        return "draw"
    if match.history.get(position_key(state), 0) >= 3:
        return "draw"
    if match.plies >= cap:
        return "cap"
    return None


def play_matches(matches, evaluator, sims, cap, batch_size, on_progress=None):
    """Opponent trees also minimize their own ordinary-outcome values.

    Results preserve legal draws versus unresolved caps. Inference is batched
    across active games and grouped by the net whose side is being evaluated.
    """
    remaining = iter(matches)
    active = []
    results = []
    rng = random.Random(0)
    while True:
        while len(active) < batch_size:
            game = next(remaining, None)
            if game is None:
                break
            game.history = {position_key(game.state): 1}
            active.append(game)
        if not active:
            break
        ongoing = []
        for game in active:
            outcome = terminal_result(game, cap)
            if outcome is None:
                ongoing.append(game)
            else:
                results.append((game, outcome))
        active = ongoing
        if active:
            searches = run_batched_mcts(
                [g.state for g in active],
                evaluator,
                sims,
                rng,
                temperatures=[0.0] * len(active),
                dirichlet_epsilon=0.0,
                invert_turns=["both"] * len(active),
                position_counts=[g.history for g in active],
                agent_colors=[g.color for g in active],
                opponent_evaluators=[g.opponent for g in active],
            )
            for game, result in zip(active, searches):
                game.state = apply_move(game.state, result.move)
                game.plies += 1
                key = position_key(game.state)
                game.history[key] = game.history.get(key, 0) + 1
        if on_progress:
            on_progress(results)
    return results


def score_interval(results):
    """95% interval over paired colors, counting unresolved caps pessimistically.

    A cap is not evidence of a draw. The displayed score gives it half a point,
    but the promotion lower bound gives it zero (upper bound gives it one).
    """
    pairs = {}
    scores = {"win": 1.0, "draw": 0.5, "loss": 0.0, "cap": 0.5}
    for match, result in results:
        pairs.setdefault(match.pair, []).append(result)

    def bound(cap_score, sign):
        values = np.array(
            [np.mean([cap_score if r == "cap" else scores[r] for r in pair]) for pair in pairs.values()]
        )
        if len(values) < 2:
            return 0.0 if sign < 0 else 1.0
        margin = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
        # Avoid unjustified certainty from a small all-identical sample.
        margin = max(margin, 1.96 * math.sqrt(0.25 / len(values)))
        return max(0.0, min(1.0, float(values.mean()) + sign * margin))

    mean = sum(scores[r] for _, r in results) / max(1, len(results))
    return mean, bound(0.0, -1), bound(1.0, 1)


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def evaluate_competitive(trainer, directory: Path):
    cfg = trainer.config
    gen = trainer.stats.generation
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "champion.pt").exists():
        trainer._save_champion(directory, gen)
    champion = trainer._load_champion_model(directory)
    opponent_gen = trainer._champion_gen
    opponents = [(f"champion-{opponent_gen}", champion)]
    seen = {model_digest(champion)}
    for i, model in enumerate(trainer._jester_opponents):
        digest = model_digest(model)
        if digest not in seen:
            opponents.append((f"historical-{i}", model))
            seen.add(digest)
    started = time.monotonic()
    was_training = trainer.model.training
    trainer.model.eval()
    evaluator = trainer._make_model_evaluator(trainer.model)
    positions = [p for p in build_eval_positions() if p.difficulty in ("opening", "middlegame")]
    # Fixed positions run once. Seeds provide genuinely different opening
    # positions, not duplicate deterministic games counted as new evidence.
    for seed in cfg.jester_eval_seeds:
        positions.extend(
            build_rotating_opening_positions(cfg.eval_rotating_openings, random.Random(seed ^ gen))
        )
    matches = []
    for opponent_name, model in opponents:
        opp_eval = trainer._make_model_evaluator(model)
        for i, position in enumerate(positions):
            for color in ("white", "black"):
                matches.append(
                    Match(position.state.copy(), color, opp_eval, f"{opponent_name}/{i}", position.difficulty)
                )
    total = len(matches)

    def progress(results):
        if trainer._mp_self_play is not None:
            trainer._mp_self_play.check_health()
            trainer._mp_self_play.drain_examples(trainer.buffer)
            for r in trainer._mp_self_play.drain_results():
                trainer._record_outcome(r.outcome, r.origin, r.resign_truth_fp, r.matchup, r.variant_outcome)
            trainer._refresh_inf_stats()
        if trainer._on_eval_progress:
            counts = {key: sum(r == key for _, r in results) for key in ("win", "loss", "draw", "cap")}
            trainer._on_eval_progress(
                len(results),
                total,
                counts["win"],
                counts["draw"] + counts["cap"],
                counts["loss"],
                {},
                current=None,
                recent=[r[0].upper() for _, r in results[-14:]],
                elapsed_s=time.monotonic() - started,
            )

    try:
        results = play_matches(
            matches, evaluator, cfg.eval_mcts_sims, cfg.eval_move_cap, cfg.jester_eval_batch_size, progress
        )
        tactics = selfmate_positions("eval")
        tactical_results = []
        for start in range(0, len(tactics), cfg.jester_eval_batch_size):
            batch = tactics[start : start + cfg.jester_eval_batch_size]
            searches = run_batched_mcts(
                [p.state for p in batch],
                evaluator,
                cfg.eval_mcts_sims,
                random.Random(0),
                temperatures=[0.0] * len(batch),
                dirichlet_epsilon=0.0,
                invert_turns=["both"] * len(batch),
                position_counts=[{position_key(p.state): 1} for p in batch],
            )
            tactical_results.extend((p, move_uci(r.move) in p.winning_moves) for p, r in zip(batch, searches))
        accuracy = sum(ok for _, ok in tactical_results) / max(1, len(tactical_results))
        by_depth = {
            depth: (
                float(np.mean([ok for p, ok in tactical_results if p.plies == depth]))
                if any(p.plies == depth for p, _ in tactical_results)
                else 0.0
            )
            for depth in (2, 4, 6)
        }
    finally:
        trainer.model.train(was_training)

    score, lower, upper = score_interval(results)
    counts = {key: sum(r == key for _, r in results) for key in ("win", "loss", "draw", "cap")}
    promote = (
        score >= cfg.eval_score_threshold and lower > 0.5 and accuracy >= cfg.jester_tactical_min_accuracy
    )
    if promote:
        trainer._save_champion(directory, gen)
        trainer._champion_model = None
        trainer._plateau_counter = 0
    else:
        trainer._plateau_counter += 1
    trainer.stats.tactical_accuracy = accuracy
    # Advance curriculum only after all three held-out depths are mastered.
    if all(value >= 0.8 for value in by_depth.values()):
        trainer.stats.curriculum_prob = max(cfg.jester_curriculum_floor, trainer.stats.curriculum_prob - 0.1)
    if trainer._mp_self_play is not None:
        trainer._mp_self_play.set_curriculum_prob(trainer.stats.curriculum_prob)
    elif trainer.engine is not None:
        trainer.engine.config.curriculum_start_prob = trainer.stats.curriculum_prob
    result = dict(
        gen=gen,
        champion_gen=trainer._champion_gen,
        opponent_gen=opponent_gen,
        gate="competitive-inverted",
        games=len(results),
        wins=counts["win"],
        draws=counts["draw"],
        losses=counts["loss"],
        caps=counts["cap"],
        score=round(score, 4),
        score_lower_95=round(lower, 4),
        score_upper_95=round(upper, 4),
        elo_diff=round(400 * math.log10(max(0.001, score) / max(0.001, 1 - score)), 1),
        new_champion=promote,
        plateau_counter=trainer._plateau_counter,
        duration_s=round(time.monotonic() - started, 1),
        tactical_accuracy=round(accuracy, 4),
        curriculum_prob=trainer.stats.curriculum_prob,
        opponents=len(opponents),
    )
    result.update({f"selfmate_{d // 2}_accuracy": float(a) for d, a in by_depth.items()})
    per_diff = {}
    for match, outcome in results:
        bucket = per_diff.setdefault(match.difficulty, dict(w=0, d=0, l=0, cap=0))
        bucket[{"win": "w", "loss": "l", "draw": "d", "cap": "cap"}[outcome]] += 1
    for name, bucket in per_diff.items():
        n = sum(bucket.values())
        result[f"score_{name}"] = (bucket["w"] + 0.5 * (bucket["d"] + bucket["cap"])) / n
        result[f"games_{name}"] = n
    path = directory / "eval.csv"
    if path.exists():
        with path.open() as f:
            columns = next(csv.reader(f))
    else:
        columns = list(result)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(result)
    result["per_diff"] = per_diff
    trainer._eval_history.append(result)
    # Full, reproducible game-level evidence, including legal draws vs caps.
    with (directory / "eval_games.jsonl").open("a") as f:
        for match, outcome in results:
            f.write(
                json.dumps(
                    dict(
                        gen=gen,
                        pair=match.pair,
                        color=match.color,
                        outcome=outcome,
                        plies=match.plies,
                        difficulty=match.difficulty,
                    )
                )
                + "\n"
            )
    return result
