"""PyTorch ChessNet — mirrors `src/game/ai/ChessNet.ts` architecturally.

Every layer here has a corresponding layer in the TF.js model. The weight
exporter in `weight_io.py` serializes this model into the browser's
`SerializedWeights` JSON so TS `ChessNet.importWeights()` loads it unchanged.

Critical parity points (must never silently drift):
  * BatchNorm eps = 1e-3 (TF.js default, NOT PyTorch's 1e-5).
  * Conv `same` padding via explicit kernel_size // 2 — stay with odd kernels.
  * Policy head flattens channels-last: permute NCHW → NHWC before flatten,
    so that policy index (rank*8 + file)*64 + to_sq matches TF.js.
  * Value head flattens the 1-channel output the same way (no-op for C=1
    but kept explicit for clarity).
  * Input is expected in NCHW: [B, NUM_PLANES, 8, 8]. The encoder emits
    [8*8*NUM_PLANES] channels-last; callers must reshape+permute.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match TF.js's `tf.layers.batchNormalization` default epsilon.
# CPUForward.ts uses this same constant.
BN_EPS = 1e-3

NUM_PLANES = 20
POLICY_SIZE = 4096
WDL_SIZE = 3
POLICY_CHANNELS = 64


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: global avg pool → dense(hidden, ReLU) → dense(filters, sigmoid)."""

    def __init__(self, filters: int, reduction: int):
        super().__init__()
        hidden = max(1, filters // reduction)
        self.fc1 = nn.Linear(filters, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, filters, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        se = x.mean(dim=(2, 3))
        se = F.relu(self.fc1(se), inplace=True)
        se = torch.sigmoid(self.fc2(se))
        return x * se.view(b, c, 1, 1)


class ResBlockSE(nn.Module):
    def __init__(self, filters: int, kernel_size: int, se_reduction: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(filters, eps=BN_EPS)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(filters, eps=BN_EPS)
        self.se = SEBlock(filters, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + x, inplace=True)


class ChessNet(nn.Module):
    """Policy + WDL value network. Returns (policy_probs, wdl_probs)."""

    def __init__(
        self,
        *,
        num_res_blocks: int,
        num_filters: int,
        kernel_size: int = 3,
        value_head_size: int = 64,
        se_reduction: int = 8,
        num_planes: int = NUM_PLANES,
        policy_size: int = POLICY_SIZE,
        wdl_size: int = WDL_SIZE,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.value_head_size = value_head_size
        self.se_reduction = se_reduction
        self.num_planes = num_planes
        self.policy_size = policy_size
        self.wdl_size = wdl_size

        pad = kernel_size // 2
        self.init_conv = nn.Conv2d(num_planes, num_filters, kernel_size, padding=pad, bias=False)
        self.init_bn = nn.BatchNorm2d(num_filters, eps=BN_EPS)
        self.res_blocks = nn.ModuleList(
            [ResBlockSE(num_filters, kernel_size, se_reduction) for _ in range(num_res_blocks)]
        )

        # Policy head: 1×1 conv (with bias, no BN) → channels-last flatten → softmax
        self.policy_conv = nn.Conv2d(num_filters, POLICY_CHANNELS, 1, bias=True)

        # Value head: 1×1 conv (no bias) → BN → ReLU → flatten → Dense(vhs, ReLU) → Dense(3, softmax)
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1, eps=BN_EPS)
        self.value_fc1 = nn.Linear(64, value_head_size)
        self.value_fc2 = nn.Linear(value_head_size, wdl_size)

    def trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared trunk (init conv + residual tower). [B, num_planes, 8, 8] -> [B, num_filters, 8, 8]."""
        h = F.relu(self.init_bn(self.init_conv(x)), inplace=True)
        for rb in self.res_blocks:
            h = rb(h)
        return h

    def heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (policy_probs, wdl_probs) from trunk features."""
        # Policy head. Permute to NHWC before flatten so index ordering matches TF.js.
        p = self.policy_conv(h)              # [B, 64, 8, 8]
        p = p.permute(0, 2, 3, 1).contiguous()  # [B, 8, 8, 64]
        p = p.view(p.size(0), -1)            # [B, 4096]
        policy = F.softmax(p, dim=1)

        v = self.value_conv(h)               # [B, 1, 8, 8]
        v = F.relu(self.value_bn(v), inplace=True)
        v = v.permute(0, 2, 3, 1).contiguous()  # [B, 8, 8, 1]
        v = v.view(v.size(0), -1)            # [B, 64]
        v = F.relu(self.value_fc1(v), inplace=True)
        v = self.value_fc2(v)
        wdl = F.softmax(v, dim=1)

        return policy, wdl

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, num_planes, 8, 8]. Returns (policy_probs [B, 4096], wdl_probs [B, 3])."""
        return self.heads(self.trunk_features(x))


class MaterialAuxHead(nn.Module):
    """KataGo-style auxiliary head: predicts material balance from trunk features.

    Lives OUTSIDE ChessNet so the browser's SerializedWeights format (which is
    TF.js-compatible and index-sensitive) stays untouched. The head consumes
    the shared trunk's output spatial features via global avg pool + linear.

    Material target is computed from the input tensor itself during training
    — no replay-buffer schema change needed.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, num_filters, 8, 8] -> [B] predicted material balance in [-1, 1]-ish."""
        pooled = h.mean(dim=(2, 3))           # [B, num_filters]
        return self.fc(pooled).squeeze(-1)    # [B]


# Piece values (K Q R B N P) matching the encoding's own/opp plane order.
# King = 0 so it cancels in the difference. Queen 9, Rook 5, Bishop 3, Knight 3, Pawn 1.
# Per-side max = 9 + 10 + 6 + 6 + 8 = 39.
_PIECE_VALUES = torch.tensor([0.0, 9.0, 5.0, 3.0, 3.0, 1.0], dtype=torch.float32)
_MAX_MATERIAL = 39.0


def material_target_from_board(x: torch.Tensor) -> torch.Tensor:
    """Compute per-position material balance from the encoded board.

    x: [B, num_planes, 8, 8] in NCHW (what ChessNet expects).
    Planes 0-5 are own pieces (K Q R B N P), 6-11 are opponent pieces.
    Returns [B] in approximately [-1, 1].
    """
    values = _PIECE_VALUES.to(x.device)
    own_counts = x[:, 0:6, :, :].sum(dim=(2, 3))    # [B, 6]
    opp_counts = x[:, 6:12, :, :].sum(dim=(2, 3))   # [B, 6]
    own_material = (own_counts * values).sum(dim=1)
    opp_material = (opp_counts * values).sum(dim=1)
    return (own_material - opp_material) / _MAX_MATERIAL


def encoded_to_nchw(encoded: torch.Tensor, num_planes: int = NUM_PLANES) -> torch.Tensor:
    """Convert a [B, 8*8*num_planes] channels-last flat tensor to [B, num_planes, 8, 8] NCHW."""
    b = encoded.shape[0]
    return encoded.view(b, 8, 8, num_planes).permute(0, 3, 1, 2).contiguous()
