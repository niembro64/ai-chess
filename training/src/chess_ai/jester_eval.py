"""Batched resistant-opponent evaluation for JESTER variants."""

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

from .engine import (
    apply_move,
    create_initial_game_state,
    get_legal_moves,
    is_in_check,
    position_key,
)
from .eval_positions import build_eval_positions, build_rotating_opening_positions
from .inverted import augmented_selfmates, selfmate_positions
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
    helper: bool = False
    start: dict = field(default_factory=dict)
    moves: list = field(default_factory=list)
    rng: object = None
    rescue_attempts: int = 0
    rescue_successes: int = 0
    exposure_attempts: int = 0
    exposure_conversions: int = 0
    immediate_check_blunders: int = 0
    candidate_exposure_attempts: int = 0
    candidate_exposure_conversions: int = 0
    candidate_rescue_attempts: int = 0
    candidate_rescue_successes: int = 0
    candidate_immediate_check_blunders: int = 0


def move_uci(move) -> str:
    def square(p):
        return f"{chr(97 + p.file)}{8 - p.rank}"

    suffix = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}.get(move.promotion, "")
    return square(move.from_pos) + square(move.to_pos) + suffix


def terminal_result(match: Match, cap: int) -> str | None:
    state = match.state
    if state.ruleset == "uncheck-v1":
        if state.status == "uncheck":
            return "win" if state.winner == match.color else "loss"
        if state.status in ("draw", "stalemate"):
            return "draw"
        if match.history.get(position_key(state), 0) >= 3:
            return "draw"
        if match.plies >= cap:
            return "cap"
        return None
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
    """Play batched resistant matches with each mover's JESTER evaluator.

    Results preserve legal draws versus unresolved caps. Inference is batched
    across games, with the actual mover's network owning every leaf in its tree.
    Explicit helper matches are diagnostics only and never enter promotion scores.
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
            game.start = game.state.to_dict()
            game.rng = random.Random(int.from_bytes(hashlib.sha256(game.pair.encode()).digest()[:8], "big"))
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
            helper_moves = {}
            for i, game in enumerate(active):
                if game.helper and game.state.currentTurn != game.color:
                    legal = get_legal_moves(game.state)
                    mates = [m for m in legal if apply_move(game.state, m).status == "checkmate"]
                    helper_moves[i] = game.rng.choice(mates or legal)
            searching = [i for i in range(len(active)) if i not in helper_moves]
            batch = [active[i] for i in searching]
            searches = run_batched_mcts(
                [g.state for g in batch],
                evaluator,
                sims,
                rng,
                temperatures=[0.0] * len(batch),
                dirichlet_epsilon=0.0,
                invert_turns=["both"] * len(batch),
                position_counts=[g.history for g in batch],
                root_evaluators=[evaluator if g.state.currentTurn == g.color else g.opponent for g in batch],
            )
            moves = dict(helper_moves)
            moves.update((i, result.move) for i, result in zip(searching, searches, strict=True))
            for i, game in enumerate(active):
                mover = game.state.currentTurn
                opponent = "black" if mover == "white" else "white"
                rescue_target = None
                if game.state.ruleset == "uncheck-v1" and is_in_check(game.state.board, opponent):
                    rescue_target = opponent
                    game.rescue_attempts += 1
                    if mover == game.color:
                        game.candidate_rescue_attempts += 1
                game.moves.append(move_uci(moves[i]))
                game.state = apply_move(game.state, moves[i])
                game.plies += 1
                if game.state.ruleset == "uncheck-v1":
                    if rescue_target is not None:
                        if game.state.status == "uncheck" and game.state.winner == rescue_target:
                            game.exposure_conversions += 1
                            if rescue_target == game.color:
                                game.candidate_exposure_conversions += 1
                        else:
                            game.rescue_successes += 1
                            if mover == game.color:
                                game.candidate_rescue_successes += 1
                    if game.state.status == "active" and is_in_check(game.state.board, mover):
                        game.exposure_attempts += 1
                        if mover == game.color:
                            game.candidate_exposure_attempts += 1
                    if game.state.status == "uncheck" and game.state.winner != mover:
                        game.immediate_check_blunders += 1
                        if mover == game.color:
                            game.candidate_immediate_check_blunders += 1
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


@dataclass(frozen=True)
class UncheckEvalStart:
    name: str
    state: object
    difficulty: str


def build_uncheck_eval_positions(count: int = 8, seed: int = 0x554E4348) -> list[UncheckEvalStart]:
    """Build fixed, reachable Uncheck openings with no normal-chess oracle.

    Each opening is produced only by Uncheck-permitted moves and remains active
    at the evaluation root. The fixed seed makes generation-to-generation
    comparisons use the same positions.
    """
    initial = create_initial_game_state("uncheck-v1")
    initial.status = "active"
    starts = [UncheckEvalStart("standard-start", initial, "standard")]
    seen = {position_key(initial)}
    targets = (4, 6, 8, 10, 12, 14, 16, 18)
    for index in range(max(0, count)):
        target_plies = targets[index % len(targets)] + 2 * (index // len(targets))
        accepted = None
        for attempt in range(128):
            rng = random.Random(seed + index * 10_007 + attempt)
            state = create_initial_game_state("uncheck-v1")
            for _ in range(target_plies):
                children = [
                    child for move in get_legal_moves(state)
                    if (child := apply_move(state, move)).status == "active"
                ]
                if not children:
                    break
                state = rng.choice(children)
            key = position_key(state)
            if state.status == "active" and key not in seen and state.fullMoveNumber > 1:
                seen.add(key)
                accepted = state
                break
        if accepted is None:
            raise RuntimeError(f"could not construct held-out Uncheck opening {index}")
        starts.append(UncheckEvalStart(f"heldout-opening-{index + 1:02d}", accepted, "heldout-opening"))
    return starts


def evaluate_uncheck(trainer, directory: Path):
    """Competitive protocol-3 evaluation using actual Uncheck winners."""
    from .mcts import run_batched_mcts
    from .uncheck import curriculum_positions

    cfg = trainer.config
    gen = trainer.stats.generation
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "champion.pt").exists():
        trainer._save_champion(directory, gen)
    champion = trainer._load_champion_model(directory)
    opponent_gen = trainer._champion_gen
    opponents = [(f"champion-{opponent_gen}", champion)]
    seen_models = {model_digest(champion)}
    for index, model in enumerate(trainer._jester_opponents):
        digest = model_digest(model)
        if digest not in seen_models:
            opponents.append((f"frozen-{index}", model))
            seen_models.add(digest)

    started = time.monotonic()
    was_training = trainer.model.training
    trainer.model.eval()
    evaluator = trainer._make_model_evaluator(trainer.model)
    opening_count = max(8, cfg.jester_eval_standard_positions * 2)
    positions = build_uncheck_eval_positions(opening_count)
    matches = []
    for opponent_name, model in opponents:
        opponent_eval = trainer._make_model_evaluator(model)
        for index, position in enumerate(positions):
            for color in ("white", "black"):
                matches.append(Match(
                    position.state.copy(), color, opponent_eval,
                    f"{opponent_name}/{index}", position.difficulty,
                ))

    total = len(matches)
    phase_started = started

    def progress(completed):
        if trainer._mp_self_play is not None:
            trainer._mp_self_play.check_health()
            trainer._mp_self_play.drain_examples(trainer.buffer)
            for result in trainer._mp_self_play.drain_results():
                trainer._record_outcome(
                    result.outcome,
                    result.origin,
                    result.resign_truth_fp,
                    result.matchup,
                    result.variant_outcome,
                )
            trainer._refresh_inf_stats()
        if trainer._on_eval_progress:
            counts = {key: sum(result == key for _, result in completed)
                      for key in ("win", "loss", "draw", "cap")}
            per_diff = {}
            for match, outcome in completed:
                bucket = per_diff.setdefault(match.difficulty, {"w": 0, "d": 0, "l": 0, "cap": 0})
                bucket[{"win": "w", "loss": "l", "draw": "d", "cap": "cap"}[outcome]] += 1
            trainer._on_eval_progress(
                len(completed), total, counts["win"], counts["draw"], counts["loss"], per_diff,
                caps=counts["cap"], phase="competitive Uncheck", current=None,
                recent=[{"win": "W", "loss": "L", "draw": "D", "cap": "C"}[result]
                        for _, result in completed[-14:]],
                elapsed_s=time.monotonic() - phase_started,
            )

    try:
        results = play_matches(
            matches, evaluator, cfg.eval_mcts_sims, cfg.eval_move_cap,
            cfg.jester_eval_batch_size, progress,
        )

        # Held-out proofs measure move recognition and end-to-end conversion,
        # but never contribute games to the competitive promotion score.
        tactics = curriculum_positions("eval")
        tactic_states = [position.state for position in tactics]
        tactic_searches = run_batched_mcts(
            tactic_states,
            evaluator,
            cfg.eval_mcts_sims,
            random.Random(0x554E4348),
            temperatures=[0.0] * len(tactics),
            dirichlet_epsilon=0.0,
            invert_turns=["both"] * len(tactics),
            position_counts=[{position_key(state): 1} for state in tactic_states],
            root_evaluators=[evaluator] * len(tactics),
        )
        tactic_hits = [move_uci(result.move) in position.winning_moves
                       for position, result in zip(tactics, tactic_searches, strict=True)]
        tactical_accuracy = sum(tactic_hits) / max(1, len(tactic_hits))
        conversion_matches = [
            Match(position.state, position.state.currentTurn,
                  trainer._make_model_evaluator(champion), f"heldout/{index}", "uncheck-tactic")
            for index, position in enumerate(tactics)
        ]
        conversions = play_matches(
            conversion_matches, evaluator, cfg.eval_mcts_sims,
            max(position.max_plies for position in tactics),
            cfg.jester_eval_batch_size,
        )
        tactical_conversion = sum(outcome == "win" for _, outcome in conversions) / max(1, len(conversions))
    finally:
        trainer.model.train(was_training)

    score, lower, upper = score_interval(results)
    counts = {key: sum(result == key for _, result in results)
              for key in ("win", "loss", "draw", "cap")}
    promote = score >= cfg.eval_score_threshold and lower > 0.5
    if promote:
        trainer._save_champion(directory, gen)
        trainer._champion_model = None
        trainer._plateau_counter = 0
    else:
        trainer._plateau_counter += 1
    trainer.stats.tactical_accuracy = tactical_accuracy
    trainer.stats.tactical_conversion = tactical_conversion

    per_diff = {}
    color_results = {
        "white": {"w": 0, "d": 0, "l": 0, "cap": 0},
        "black": {"w": 0, "d": 0, "l": 0, "cap": 0},
    }
    for match, outcome in results:
        bucket = per_diff.setdefault(match.difficulty, {"w": 0, "d": 0, "l": 0, "cap": 0})
        key = {"win": "w", "loss": "l", "draw": "d", "cap": "cap"}[outcome]
        bucket[key] += 1
        color_results[match.color][key] += 1

    white_wins = sum(match.state.status == "uncheck" and match.state.winner == "white"
                     for match, _ in results)
    black_wins = sum(match.state.status == "uncheck" and match.state.winner == "black"
                     for match, _ in results)
    result = dict(
        gen=gen,
        champion_gen=trainer._champion_gen,
        opponent_gen=opponent_gen,
        gate="competitive-uncheck-v1",
        games=len(results),
        wins=counts["win"],
        draws=counts["draw"],
        losses=counts["loss"],
        caps=counts["cap"],
        score=round(score, 4),
        score_lower_bound=round(lower, 4),
        score_lower_95=round(lower, 4),
        score_upper_95=round(upper, 4),
        elo_diff=round(400 * math.log10(max(0.001, score) / max(0.001, 1 - score)), 1),
        new_champion=promote,
        plateau_counter=trainer._plateau_counter,
        duration_s=round(time.monotonic() - started, 1),
        average_plies=round(sum(match.plies for match, _ in results) / max(1, len(results)), 1),
        white_wins=white_wins,
        black_wins=black_wins,
        challenger_white=color_results["white"],
        challenger_black=color_results["black"],
        rescue_attempts=sum(match.rescue_attempts for match, _ in results),
        rescue_successes=sum(match.rescue_successes for match, _ in results),
        candidate_rescue_attempts=sum(match.candidate_rescue_attempts for match, _ in results),
        candidate_rescue_successes=sum(match.candidate_rescue_successes for match, _ in results),
        immediate_check_blunders=sum(match.immediate_check_blunders for match, _ in results),
        candidate_immediate_check_blunders=sum(
            match.candidate_immediate_check_blunders for match, _ in results
        ),
        forced_check_attempts=sum(match.candidate_exposure_attempts for match, _ in results),
        forced_check_conversions=sum(match.candidate_exposure_conversions for match, _ in results),
        tactical_accuracy=round(tactical_accuracy, 4),
        tactical_conversion=round(tactical_conversion, 4),
        opponents=len(opponents),
        per_diff=per_diff,
    )
    for name, bucket in per_diff.items():
        sample_count = sum(bucket.values())
        column = name.replace("-", "_")
        result[f"score_{column}"] = (
            bucket["w"] + 0.5 * (bucket["d"] + bucket["cap"])
        ) / max(1, sample_count)
        result[f"games_{column}"] = sample_count

    trainer._eval_history.append(result)
    csv_row = {key: value for key, value in result.items()
               if key not in ("per_diff", "challenger_white", "challenger_black")}
    path = directory / "eval.csv"
    if path.exists():
        with path.open() as file:
            columns = next(csv.reader(file))
    else:
        columns = list(csv_row)
    new_file = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(csv_row)

    with (directory / "eval_games.jsonl").open("a") as file:
        for match, outcome in results:
            file.write(json.dumps(dict(
                gen=gen,
                pair=match.pair,
                candidate_color=match.color,
                outcome=outcome,
                actual_winner=match.state.winner,
                plies=match.plies,
                start_type=match.difficulty,
                start=match.start,
                moves=match.moves,
                final_status=match.state.status,
                rescue_attempts=match.rescue_attempts,
                rescue_successes=match.rescue_successes,
                immediate_check_blunders=match.immediate_check_blunders,
                candidate_exposure_attempts=match.candidate_exposure_attempts,
                candidate_exposure_conversions=match.candidate_exposure_conversions,
            )) + "\n")
    with (directory / "diagnostics.jsonl").open("a") as file:
        for (match, outcome), hit in zip(conversions, tactic_hits, strict=True):
            file.write(json.dumps(dict(
                gen=gen,
                suite="heldout-uncheck",
                pair=match.pair,
                first_move_correct=hit,
                outcome=outcome,
                start=match.start,
                moves=match.moves,
            )) + "\n")
    return result


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
    regular = [p for p in build_eval_positions() if p.difficulty in ("opening", "middlegame")]
    full_eval = gen > 0 and cfg.jester_full_eval_every > 0 and gen % cfg.jester_full_eval_every == 0
    positions = regular if full_eval else [regular[i] for i in np.linspace(
        0, len(regular) - 1, min(len(regular), cfg.jester_eval_standard_positions), dtype=int)]
    # Fixed positions run once. Seeds provide genuinely different opening
    # positions, not duplicate deterministic games counted as new evidence.
    for seed in cfg.jester_eval_seeds:
        positions.extend(
            build_rotating_opening_positions(10 if full_eval else cfg.eval_rotating_openings, random.Random(seed ^ gen))
        )
    # Resistant near-goal games supply measurable competitive signal alongside
    # ordinary starts. Fixed held-out bases are separate from training geometry.
    positions.extend(selfmate_positions("eval"))
    matches = []
    for opponent_name, model in opponents:
        opp_eval = trainer._make_model_evaluator(model)
        for i, position in enumerate(positions):
            for color in ("white", "black"):
                matches.append(
                    Match(position.state.copy(), color, opp_eval, f"{opponent_name}/{i}", position.difficulty)
                )
    total = len(matches)
    phase = "competitive"
    phase_started = started

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
                counts["draw"],
                counts["loss"],
                {},
                caps=counts["cap"],
                phase=phase,
                current=None,
                recent=[r[0].upper() for _, r in results[-14:]],
                elapsed_s=time.monotonic() - phase_started,
            )

    try:
        results = play_matches(
            matches, evaluator, max(256, cfg.eval_mcts_sims) if full_eval else cfg.eval_mcts_sims,
            max(300, cfg.eval_move_cap) if full_eval else cfg.eval_move_cap, cfg.jester_eval_batch_size, progress
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
            tactical_results.extend((p, move_uci(r.move) in p.winning_moves) for p, r in zip(batch, searches, strict=True))
        accuracy = sum(ok for _, ok in tactical_results) / max(1, len(tactical_results))
        by_depth = {
            depth: (
                float(np.mean([ok for p, ok in tactical_results if p.plies == depth]))
                if any(p.plies == depth for p, _ in tactical_results)
                else 0.0
            )
            for depth in (2, 4, 6)
        }
        geometric = augmented_selfmates("eval")
        geometric = [random.Random(910 + depth).sample([p for p in geometric if p.plies == depth], 6)
                     for depth in (2, 4, 6)]
        conversion_positions = list(tactics) + [p for group in geometric for p in group]
        total, phase = len(conversion_positions), "full tactical conversion"
        phase_started = time.monotonic()
        conversions = play_matches(
            [Match(p.state, p.state.currentTurn, evaluator, f"conversion/{i}", p.difficulty)
             for i, p in enumerate(conversion_positions)],
            evaluator, cfg.eval_mcts_sims, min(12, cfg.eval_move_cap), cfg.jester_eval_batch_size, progress)
        conversion = sum(outcome == "win" for _, outcome in conversions) / max(1, len(conversions))
        geometry_conversion = sum(outcome == "win" for _, outcome in conversions[len(tactics):]) / 18
        # Same fixed starts, colors and random streams for the candidate and
        # champion. These cooperative results are diagnostic, not Elo or gate evidence.
        helper_results = {}
        helper_games = []
        for name, model in (("candidate", trainer.model), ("champion", champion)):
            helper_eval = trainer._make_model_evaluator(model)
            total, phase = 8, f"helper diagnostic: {name}"
            phase_started = time.monotonic()
            played = play_matches(
                [Match(regular[i].state.copy(), color, helper_eval, f"helper/{i}/{color}", "helper", helper=True)
                 for i in (0, 9, 19, 29) if i < len(regular) for color in ("white", "black")],
                helper_eval, min(96, cfg.eval_mcts_sims), min(120, cfg.eval_move_cap),
                cfg.jester_eval_batch_size, progress)
            helper_results[name] = {outcome: sum(r == outcome for _, r in played)
                                    for outcome in ("win", "loss", "draw", "cap")}
            helper_games.extend((name, game, outcome) for game, outcome in played)
    finally:
        trainer.model.train(was_training)

    score, lower, upper = score_interval(results)
    counts = {key: sum(r == key for _, r in results) for key in ("win", "loss", "draw", "cap")}
    promote = (
        score >= cfg.eval_score_threshold and lower > 0.5
        and accuracy >= cfg.jester_tactical_min_accuracy and conversion >= cfg.jester_tactical_min_accuracy
    )
    if promote:
        trainer._save_champion(directory, gen)
        trainer._champion_model = None
        trainer._plateau_counter = 0
    else:
        trainer._plateau_counter += 1
    trainer.stats.tactical_accuracy = accuracy
    trainer.stats.tactical_conversion = conversion
    # Never anneal from a narrow first-move quiz. Reduce helper availability only
    # after a promotion, full conversions and actual decisive ordinary play.
    ordinary_decisive = sum(outcome in ("win", "loss") for game, outcome in results
                            if not game.difficulty.startswith("selfmate"))
    if promote and conversion >= .8 and ordinary_decisive >= 20 and trainer.stats.helper_prob > .10:
        trainer.stats.helper_prob = max(.10, trainer.stats.helper_prob - .05)
    if trainer._mp_self_play is not None:
        trainer._mp_self_play.set_helper_prob(trainer.stats.helper_prob)
    elif trainer.engine is not None:
        trainer.engine.config.helper_start_prob = trainer.stats.helper_prob
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
        tactical_conversion=round(conversion, 4),
        geometry_conversion=round(geometry_conversion, 4),
        helper_results=json.dumps(helper_results, sort_keys=True),
        helper_prob=trainer.stats.helper_prob,
        full_eval=full_eval,
        decisive_games=counts["win"] + counts["loss"],
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
                        start=match.start,
                        moves=match.moves,
                        final_status=match.state.status,
                    )
                )
                + "\n"
            )
    with (directory / "diagnostics.jsonl").open("a") as f:
        for suite, game, outcome in [("conversion", g, r) for g, r in conversions] + helper_games:
            f.write(json.dumps(dict(gen=gen, suite=suite, pair=game.pair, color=game.color,
                                    start=game.start, moves=game.moves, outcome=outcome)) + "\n")
    return result
