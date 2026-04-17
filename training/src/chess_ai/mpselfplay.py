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
    games_per_worker: int = 16
    mcts_simulations: int = 25
    # Inference batching: collect requests for up to `batch_wait_ms`
    # (or until enough arrive) before dispatching to the GPU. Small numbers
    # trade latency for throughput.
    batch_wait_ms: float = 5.0
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
) -> None:
    """Own the GPU, serve batched inference requests.

    Protocol:
      request_q item:  (worker_id: int, boards_ndarray: np.ndarray[B, 8*8*20])
      response_qs[i] item: (policies_ndarray, values_ndarray)
    """
    torch.set_num_threads(1)          # inference server only; workers don't use torch ops
    device = torch.device(device_str)
    model = arch.build().to(device).eval()
    model.load_state_dict(initial_state_dict)

    wait_s = batch_wait_ms / 1000.0

    # Track some stats for diagnosis (can be read by parent via a different
    # channel if wanted; for now just left accessible for future extension).
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

        pending: list[tuple[int, np.ndarray]] = [first]
        deadline = time.monotonic() + wait_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pending.append(request_q.get(timeout=remaining))
            except Empty:
                break

        # Concatenate into one big batch, remember per-worker offsets.
        slices: list[tuple[int, int, int]] = []  # (worker_id, start, end)
        offset = 0
        boards_blocks = []
        for worker_id, boards in pending:
            n = boards.shape[0]
            slices.append((worker_id, offset, offset + n))
            offset += n
            boards_blocks.append(boards)
        merged = np.concatenate(boards_blocks, axis=0)

        # Forward on GPU.
        with torch.no_grad():
            x_flat = torch.from_numpy(merged).to(device)
            x = encoded_to_nchw(x_flat, NUM_PLANES)
            policy, wdl = model(x)
        pol = policy.cpu().numpy()
        w = wdl.cpu().numpy()
        values = (w[:, 0] - w[:, 2]).astype(np.float32)

        # Fan the results back out to each worker's response queue.
        for worker_id, start, end in slices:
            response_qs[worker_id].put((pol[start:end], values[start:end]))


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
) -> None:
    """Run a self-play loop; NN evals go through the shared inference server."""
    torch.set_num_threads(1)

    def remote_evaluator(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        request_q.put((worker_id, boards))
        return response_q.get()

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
            rewards=config.rewards,
        ),
        rng=rng,
    )

    while not stop_event.is_set():
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
                ),
                daemon=True,
            )
            p.start()
            self._worker_procs.append(p)

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
