"""Toy — the teaching-sized network, as a first-class citizen of the
main training pipeline.

Same skeleton as Sage (shared trunk, policy head, WDL value head), a
fraction of the size, and a deliberately minimal 6-plane input that is
blind to castling rights / en passant / move clocks. Because ToyNet
implements the same `trunk_features()` / `heads()` contract as
ChessNet and carries `num_planes`, the Trainer, self-play engine,
dashboard, eval gating, mirror augmentation, resign logic, tablebase
adjudication — the whole helper suite — work on it unchanged. The only
Toy-specific pieces are the encoder below and the browser JSON format
("toy-v1", consumed by src/game/ai/ToyNet.ts).
"""

from __future__ import annotations

import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import ChessGameState

TOY_NUM_PLANES = 6
PIECE_CHANNEL = {"pawn": 0, "knight": 1, "bishop": 2, "rook": 3, "queen": 4, "king": 5}
POLICY_SIZE = 4096

TOY_FILTERS = 32
TOY_BLOCKS = 3
TOY_POLICY_CH = 4
TOY_VALUE_CH = 2
TOY_VALUE_HIDDEN = 64


def encode_toy(state: ChessGameState) -> np.ndarray:
    """Flat [8*8*6] float32, channels-last: idx = (rank*8 + file)*6 + ch.

    Mover's perspective: own pieces +1, opponent -1; board rotated 180°
    (rank AND file) when black moves — identical convention to Sage's
    20-plane encoder, so positions/moves stay mentally comparable.
    Mirrors src/game/ai/ToyNet.ts encodeToyBoard byte for byte.
    """
    x = np.zeros(8 * 8 * TOY_NUM_PLANES, dtype=np.float32)
    white_to_move = state.currentTurn == "white"
    for r in range(8):
        for f in range(8):
            p = state.board[r][f]
            if p is None:
                continue
            rr, ff = (r, f) if white_to_move else (7 - r, 7 - f)
            sign = 1.0 if (p.color == "white") == white_to_move else -1.0
            x[(rr * 8 + ff) * TOY_NUM_PLANES + PIECE_CHANNEL[p.type]] = sign
    return x


class ToyNet(nn.Module):
    """Policy + WDL value network, ChessNet-compatible surface.

    3 BN-free residual blocks x 32 filters. `heads()` returns
    (policy_probs [B, 4096], wdl_probs [B, 3]) exactly like ChessNet,
    so Trainer.train_step / evaluators / eval gating need no branches.
    Value parameters are named value_* so the value-head warmup freeze
    matches them too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_planes = TOY_NUM_PLANES
        self.num_res_blocks = TOY_BLOCKS
        self.num_filters = TOY_FILTERS

        self.conv_in = nn.Conv2d(TOY_NUM_PLANES, TOY_FILTERS, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(TOY_BLOCKS):
            self.blocks.append(nn.ModuleDict({
                "conv1": nn.Conv2d(TOY_FILTERS, TOY_FILTERS, 3, padding=1),
                "conv2": nn.Conv2d(TOY_FILTERS, TOY_FILTERS, 3, padding=1),
            }))
        self.policy_conv = nn.Conv2d(TOY_FILTERS, TOY_POLICY_CH, 1)
        self.policy_fc = nn.Linear(8 * 8 * TOY_POLICY_CH, POLICY_SIZE)
        self.value_conv = nn.Conv2d(TOY_FILTERS, TOY_VALUE_CH, 1)
        self.value_fc1 = nn.Linear(8 * 8 * TOY_VALUE_CH, TOY_VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(TOY_VALUE_HIDDEN, 3)

    @staticmethod
    def _flatten_hwc(t: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC -> flat so the browser (channels-last) flattens in
        # exactly the same order. THE load-bearing transpose.
        return t.permute(0, 2, 3, 1).reshape(t.shape[0], -1)

    def trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv_in(x))
        for b in self.blocks:
            y = b["conv2"](F.relu(b["conv1"](h)))
            h = F.relu(h + y)
        return h

    def heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._flatten_hwc(F.relu(self.policy_conv(h)))
        policy = F.softmax(self.policy_fc(p), dim=-1)
        v = self._flatten_hwc(F.relu(self.value_conv(h)))
        wdl = F.softmax(self.value_fc2(F.relu(self.value_fc1(v))), dim=-1)
        return policy, wdl

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(self.trunk_features(x))

    # -- pipeline integration hooks ---------------------------------------

    def arch_dict(self) -> dict:
        """Self-describing checkpoint metadata (family picks the class
        on champion reload)."""
        return {"family": "toy"}

    def export_browser_json(self) -> dict:
        """"toy-v1" browser format: named tensors in fixed order, fp16
        base64, conv filters in TF layout [h,w,in,out], linear [in,out].
        Consumed by src/game/ai/ToyNet.ts."""
        tensors: list[tuple[str, np.ndarray]] = []

        def conv(name: str, m: nn.Conv2d) -> None:
            tensors.append((f"{name}.w", m.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)))
            tensors.append((f"{name}.b", m.bias.detach().cpu().numpy()))

        def fc(name: str, m: nn.Linear) -> None:
            tensors.append((f"{name}.w", m.weight.detach().cpu().numpy().T))
            tensors.append((f"{name}.b", m.bias.detach().cpu().numpy()))

        conv("conv_in", self.conv_in)
        for i, b in enumerate(self.blocks):
            conv(f"block{i}.conv1", b["conv1"])
            conv(f"block{i}.conv2", b["conv2"])
        conv("policy_conv", self.policy_conv)
        fc("policy_fc", self.policy_fc)
        conv("value_conv", self.value_conv)
        fc("value_fc1", self.value_fc1)
        fc("value_fc2", self.value_fc2)

        return {
            "kind": "toy-v1",
            "config": {
                "numPlanes": TOY_NUM_PLANES,
                "numFilters": TOY_FILTERS,
                "numResBlocks": TOY_BLOCKS,
                "policyChannels": TOY_POLICY_CH,
                "valueChannels": TOY_VALUE_CH,
                "valueHidden": TOY_VALUE_HIDDEN,
            },
            "names": [n for n, _ in tensors],
            "shapes": [list(a.shape) for _, a in tensors],
            "data": [
                base64.b64encode(a.astype("<f2").tobytes()).decode("ascii")
                for _, a in tensors
            ],
        }
