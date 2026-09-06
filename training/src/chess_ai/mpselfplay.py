"""Multi-process self-play with a central GPU inference server.

Architecture
------------

  [ trainer (main) ]   ──── state_dict ───▶  [ inference server ]
        ▲                                        ▲      │
        │  TrainingExample                       │      │ policies+values
        │                            (boards)    │      ▼
        │                                        │   [ workers × N ]
        └────────────────────────────────────────┘
        example_q                            request_q / response_q

Why this shape
--------------
Python's GIL plus the per-step MCTS tree walking is the bottleneck even
after the hot-path optimizations. In single-process mode the GPU sits at
0% util because one Python process can only feed it so fast. With N
CPU-bound worker processes doing self-play in parallel and a dedicated
GPU process batching their eval requests, we:

  * scale MCTS throughput roughly linearly with cores, and
  * combine many per-worker batches into one big GPU dispatch so the
    3090 actually has something to chew on.

The canonical model lives in the trainer process (gradient updates).
Every `weight_broadcast_every` gradient steps, the trainer pushes a
fresh `state_dict()` over `weights_q` and the inference server hot-swaps.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from .model import NUM_PLANES, ChessNet, encoded_to_nchw
from .rewards import DEFAULT_REWARD_WEIGHTS, RewardWeights
from .selfplay import (
    GameResult,
    ReplayBuffer,
    SelfPlayConfig,
    SelfPlayEngine,
    TrainingExample,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiprocessingConfig:
    num_workers: int = 4
    jester_mode: bool = False
    jester_selfplay_prob: float = 0.75
    opponent_checkpoints: tuple[str, ...] = ()
    curriculum_start_prob: float = 0.0
    standard_move_cap: int = 300
    games_per_worker: int = 16
    mcts_simulations: int = 25
    # Inference batching: collect requests for up to `batch_wait_ms`
    # (or until enough arrive) before dispatching to the GPU. Small numbers
    # trade latency for throughput.
    batch_wait_ms: float = 5.0
    # Self-play scheduling knobs — mirrored to each worker's SelfPlayConfig.
    endgame_start_prob: float = 0.0
    random_start_prob: float = 0.3
    temperature_threshold_plies: int = 15
    # MCTS hyperparams — applied in each worker via set_mcts_params() at
    # startup. None = use the module defaults in mcts.py.
    c_puct: float | None = None
    dirichlet_alpha: float | None = None
    dirichlet_epsilon: float | None = None
    # First-Play Urgency reduction — see TrainConfig.fpu_reduction.
    fpu_reduction: float | None = None
    # Syzygy tablebase directory (passed to every worker since worker
    # processes don't inherit parent globals under "spawn" start method).
    # None = disabled.
    syzygy_path: str | None = None
    syzygy_max_pieces: int = 5
    # Per-ply decay on the value target. <1.0 teaches the value head
    # to prefer fast wins (see TrainConfig.value_ply_decay for why).
    value_ply_decay: float = 1.0
    # Soften NN policy priors in self-play MCTS (>1.0 = flatter prior).
    # See SelfPlayConfig.policy_softening_temperature.
    policy_softening_temperature: float = 1.0
    # Per-sample policy-loss weight for TB-adjudicated games. See
    # SelfPlayConfig.tb_policy_weight. 1.0 = no discount.
    tb_policy_weight: float = 1.0
    # Resignation knobs — see TrainConfig.resign_*. Mirrored to each
    # worker's SelfPlayConfig.
    resign_threshold: float = -0.85
    resign_disabled_prob: float = 0.10
    resign_min_plies: int = 20
    # Queue size bounds. Bounded queues apply backpressure if one side
    # pulls ahead; None = unbounded.
    request_q_maxsize: int = 0
    response_q_maxsize: int = 4
    example_q_maxsize: int = 4096
    rewards: RewardWeights = field(
        default_factory=lambda: RewardWeights(**DEFAULT_REWARD_WEIGHTS.__dict__)
    )


@dataclass
class ModelArch:
    """Pickle-friendly snapshot of a ChessNet's architecture so child processes
    can rebuild the exact same model topology before loading weights."""

    num_res_blocks: int
    num_filters: int
    kernel_size: int
    value_head_size: int
    se_reduction: int

    @classmethod
    def from_model(cls, model: ChessNet) -> "ModelArch":
        return cls(
            num_res_blocks=model.num_res_blocks,
            num_filters=model.num_filters,
            kernel_size=model.kernel_size,
            value_head_size=model.value_head_size,
            se_reduction=model.se_reduction,
        )

    def build(self) -> ChessNet:
        return ChessNet(
            num_res_blocks=self.num_res_blocks,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            value_head_size=self.value_head_size,
            se_reduction=self.se_reduction,
        )


# ---------------------------------------------------------------------------
# Inference server (runs in its own process)
# ---------------------------------------------------------------------------


def _inference_server_main(
    arch: ModelArch,
    initial_state_dict: dict,
    device_str: str,
    request_q: "mp.Queue",
    response_qs: list,
    weights_q: "mp.Queue",
    stop_event: Any,
    batch_wait_ms: float,
    opponent_checkpoints: tuple[str, ...],
    inf_stats: Any,  # mp.Array('q', 4) — see snapshot_inf_stats()
) -> None:
    """Own the GPU, serve batched inference requests.

    Protocol:
      request_q item:  (worker_id: int, model_id: int, boards_ndarray: np.ndarray[B, 8*8*20])
      response_qs[i] item: (policies_ndarray, values_ndarray)
    """
    torch.set_num_threads(1)          # inference server only; workers don't use torch ops
    device = torch.device(device_str)
    model = arch.build().to(device).eval()
    model.load_state_dict(initial_state_dict)
    opponents = []
    for path in opponent_checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        opponent = arch.build().to(device).eval()
        opponent.load_state_dict(checkpoint["model_state_dict"])
        opponents.append(opponent)

    # torch.compile the inference forward pass on CUDA. Inference batch
    # sizes vary (sum of pending requests), so use dynamic=True to avoid
    # a recompile per shape. We keep `model` as the raw nn.Module — that
    # way load_state_dict still works directly, and the compiled graph
    # picks up new weights through the module's parameter references.
    # Falls back silently if torch.compile is missing or fails.
    forward_fn = model
    if (
        device.type == "cuda"
        and os.environ.get("CHESS_AI_NO_COMPILE") != "1"
        and hasattr(torch, "compile")
    ):
        try:
            forward_fn = torch.compile(model, dynamic=True)
            logging.getLogger("chess_ai.train").info(
                "torch.compile: inference server forward compiled (dynamic=True)"
            )
        except Exception as e:
            logging.getLogger("chess_ai.train").warning(
                "torch.compile failed (inference): %s; falling back to eager", e
            )

    wait_s = batch_wait_ms / 1000.0

    while not stop_event.is_set():
        # Opportunistic weight swap (non-blocking).
        try:
            new_state = weights_q.get_nowait()
            model.load_state_dict(new_state)
            del new_state
        except Empty:
            pass

        # Block for the first request, then briefly drain additional ones
        # before dispatching a merged batch.
        try:
            first = request_q.get(timeout=0.05)
        except Empty:
            continue
        first_recv_t = time.monotonic()

        pending = [first]
        deadline = first_recv_t + wait_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pending.append(request_q.get(timeout=remaining))
            except Empty:
                break

        # Batch independently for each immutable opponent/current model.
        # Each worker has one outstanding request, so its reply queue is unambiguous.
        grouped = {}
        for worker_id, model_id, boards in pending:
            grouped.setdefault(model_id, []).append((worker_id, boards))
        for model_id, requests in grouped.items():
            merged = np.concatenate([b for _, b in requests], axis=0)
            wait_us = int((time.monotonic() - first_recv_t) * 1e6)
            with inf_stats.get_lock():
                inf_stats[0] += 1
                inf_stats[1] += merged.shape[0]
                inf_stats[2] = max(inf_stats[2], merged.shape[0])
                inf_stats[3] += wait_us
            fn = forward_fn if model_id == 0 else opponents[model_id - 1]
            with torch.no_grad():
                x = encoded_to_nchw(torch.from_numpy(merged).to(device), NUM_PLANES)
                policy, wdl = fn(x)
            pol = policy.cpu().numpy()
            w = wdl.cpu().numpy()
            values = (w[:, 0] - w[:, 2]).astype(np.float32)
            offset = 0
            for worker_id, boards in requests:
                n = boards.shape[0]
                response_qs[worker_id].put((pol[offset:offset+n], values[offset:offset+n]))
                offset += n


# ---------------------------------------------------------------------------
# Worker (runs in its own process)
# ---------------------------------------------------------------------------


def _worker_main(
    worker_id: int,
    config: MultiprocessingConfig,
    request_q: "mp.Queue",
    response_q: "mp.Queue",
    example_q: "mp.Queue",
    results_q: "mp.Queue",
    stop_event: Any,
    seed: int,
    curriculum_prob: Any,
) -> None:
    """Run a self-play loop; NN evals go through the shared inference server."""
    torch.set_num_threads(1)

    # Every worker is its own Python process, so it has its own copy of the
    # mcts module globals. Re-apply the overrides here.
    from .mcts import set_mcts_params
    set_mcts_params(
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        fpu_reduction=config.fpu_reduction,
    )
    # Same reason for the Syzygy tablebase: each worker must open it itself.
    if config.syzygy_path and not config.jester_mode:
        from . import tablebase
        tablebase.open_tablebase(config.syzygy_path, config.syzygy_max_pieces)

    def evaluator_for(model_id):
        def evaluate(boards):
            request_q.put((worker_id, model_id, boards))
            while not stop_event.is_set():
                try:
                    return response_q.get(timeout=1.0)
                except Empty:
                    continue
            raise SystemExit(0)
        return evaluate

    remote_evaluator = evaluator_for(0)
    opponents = tuple(evaluator_for(i + 1) for i in range(len(config.opponent_checkpoints)))

    rng = random.Random(seed)

    def push_example(ex: TrainingExample) -> None:
        # Block if trainer falls behind — applies backpressure.
        example_q.put(ex)

    engine = SelfPlayEngine(
        evaluator=remote_evaluator,
        example_sink=push_example,
        config=SelfPlayConfig(
            num_concurrent_games=config.games_per_worker,
            mcts_simulations=config.mcts_simulations,
            invert_agent_selection=config.jester_mode,
            frozen_evaluators=opponents,
            opponent_seeks_loss=True,
            agent_selfplay_prob=config.jester_selfplay_prob,
            curriculum_start_prob=config.curriculum_start_prob,
            standard_move_cap=config.standard_move_cap,
            spar_temperature=0.0,
            spar_random_prob=0.0,
            spar_accept_mate_prob=0.0,
            endgame_start_prob=config.endgame_start_prob,
            random_start_prob=config.random_start_prob,
            temperature_threshold_plies=config.temperature_threshold_plies,
            rewards=config.rewards,
            value_ply_decay=config.value_ply_decay,
            policy_softening_temperature=config.policy_softening_temperature,
            tb_policy_weight=config.tb_policy_weight,
            resign_threshold=config.resign_threshold,
            resign_disabled_prob=config.resign_disabled_prob,
            resign_min_plies=config.resign_min_plies,
        ),
        rng=rng,
    )

    while not stop_event.is_set():
        engine.config.curriculum_start_prob = curriculum_prob.value
        finished = engine.step()
        # Surface per-game outcomes back to the trainer so the dashboard's
        # outcomes panel shows real W/B/D/cap counts instead of zeros.
        for result in finished:
            results_q.put(result)


# ---------------------------------------------------------------------------
# Orchestrator (runs inside trainer process)
# ---------------------------------------------------------------------------


class MultiprocessingSelfPlay:
    """Spawns + manages the inference server and N workers.

    Usage from the trainer:

        mp_sp = MultiprocessingSelfPlay(arch, model.state_dict(), device, cfg)
        mp_sp.start()
        try:
            while training:
                drained = mp_sp.drain_examples(replay_buffer)
                # ... gradient step, etc. ...
                if should_broadcast:
                    mp_sp.broadcast_weights(model.state_dict())
        finally:
            mp_sp.stop()
    """

    def __init__(
        self,
        arch: ModelArch,
        initial_state_dict: dict,
        device: torch.device,
        config: MultiprocessingConfig,
        seed: int = 0,
    ):
        self.arch = arch
        self.initial_state_dict = initial_state_dict
        self.device = device
        self.config = config
        self.seed = seed

        # `spawn` is required for CUDA-safe child processes on Linux/macOS.
        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._curriculum_prob = self._ctx.Value("d", config.curriculum_start_prob)

        self._request_q: mp.Queue = self._ctx.Queue(maxsize=config.request_q_maxsize)
        self._response_qs: list[mp.Queue] = [
            self._ctx.Queue(maxsize=config.response_q_maxsize)
            for _ in range(config.num_workers)
        ]
        self._example_q: mp.Queue = self._ctx.Queue(maxsize=config.example_q_maxsize)
        # Worker-reported GameResult stream (W/B/D/cap, move counts). Bounded
        # so that if the trainer ever stops draining, workers block rather
        # than leaking memory.
        self._results_q: mp.Queue = self._ctx.Queue(maxsize=config.example_q_maxsize)
        # weights_q: bounded to 1 so stale updates are auto-evicted.
        self._weights_q: mp.Queue = self._ctx.Queue(maxsize=1)
        # Inference-server stats. Four signed int64 counters, lock-protected
        # by mp.Array's built-in lock. Slots:
        #   0 = n_dispatches            (cumulative)
        #   1 = sum_batch_size          (cumulative — divide by [0] for avg)
        #   2 = peak_batch_size         (running max, never reset)
        #   3 = sum_wait_us             (cumulative wait between first req and
        #                                dispatch — divide by [0] for avg ms)
        # Read+reset via snapshot_inf_stats() to compute window-rate metrics.
        self._inf_stats = self._ctx.Array("q", 4)

        self._inference_proc = None
        self._worker_procs: list[mp.Process] = []

    def start(self) -> None:
        self._inference_proc = self._ctx.Process(
            target=_inference_server_main,
            args=(
                self.arch,
                self.initial_state_dict,
                str(self.device),
                self._request_q,
                self._response_qs,
                self._weights_q,
                self._stop_event,
                self.config.batch_wait_ms,
                self.config.opponent_checkpoints,
                self._inf_stats,
            ),
            daemon=True,
        )
        self._inference_proc.start()

        for wid in range(self.config.num_workers):
            p = self._ctx.Process(
                target=_worker_main,
                args=(
                    wid,
                    self.config,
                    self._request_q,
                    self._response_qs[wid],
                    self._example_q,
                    self._results_q,
                    self._stop_event,
                    self.seed + wid + 1,
                    self._curriculum_prob,
                ),
                daemon=True,
            )
            p.start()
            self._worker_procs.append(p)

    def check_health(self) -> None:
        for process in [self._inference_proc, *self._worker_procs]:
            if process is not None and process.exitcode is not None:
                raise RuntimeError(f"Self-play process {process.name} exited with {process.exitcode}")

    def set_curriculum_prob(self, probability: float) -> None:
        self._curriculum_prob.value = probability

    def drain_examples(self, buffer: ReplayBuffer, max_drain: int = 4096) -> int:
        """Pull completed TrainingExamples off the queue into the buffer.

        Returns the number pulled. Non-blocking beyond the first get — if the
        queue is empty the call returns immediately.
        """
        pulled = 0
        while pulled < max_drain:
            try:
                ex = self._example_q.get_nowait()
            except Empty:
                break
            buffer.add(ex)
            pulled += 1
        return pulled

    def drain_results(self, max_drain: int = 4096) -> list[GameResult]:
        """Pull any GameResults workers have posted since the last drain."""
        results: list[GameResult] = []
        for _ in range(max_drain):
            try:
                results.append(self._results_q.get_nowait())
            except Empty:
                break
        return results

    def broadcast_weights(self, state_dict: dict) -> None:
        """Push fresh weights to the inference server. Only the latest is
        retained — if a previous broadcast is still in flight it's discarded."""
        # Drain stale updates (keep queue at ≤1 entry).
        try:
            self._weights_q.get_nowait()
        except Empty:
            pass
        # CPU copies — the inference server's process can't share CUDA tensors
        # with the trainer's process without going through NCCL/IPC, and the
        # size is tiny (~12 MB for 3M params) so just ship CPU tensors.
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        self._weights_q.put(cpu_state)

    def snapshot_inf_stats(self) -> dict:
        """Read the inference server's cumulative counters atomically.

        Returns a dict with keys: dispatches, sum_batch, peak_batch,
        sum_wait_us. Reading is cheap (lock + 4 int64 reads); the trainer
        calls this once per main-loop iter and computes deltas itself.
        """
        with self._inf_stats.get_lock():
            return {
                "dispatches": self._inf_stats[0],
                "sum_batch": self._inf_stats[1],
                "peak_batch": self._inf_stats[2],
                "sum_wait_us": self._inf_stats[3],
            }

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        for p in self._worker_procs:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
        if self._inference_proc is not None:
            self._inference_proc.join(timeout=timeout)
            if self._inference_proc.is_alive():
                self._inference_proc.terminate()

    @property
    def example_queue_size(self) -> int:
        # Best-effort; on some platforms .qsize() is not implemented.
        try:
            return self._example_q.qsize()
        except NotImplementedError:
            return -1
