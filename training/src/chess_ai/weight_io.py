"""Weight import/export for round-tripping PyTorch ChessNet ↔ browser TF.js ChessNet.

The output format is the `SerializedWeights` type defined in
`src/game/ai/ChessNet.ts`:

    { "config": NetConfig,
      "shapes": number[][],
      "data":   number[][] }

Weight *ordering* must match the TF.js model's `model.weights` traversal order,
since the browser's `importWeights()` simply zips shapes/data with
`model.setWeights()`. Get the order wrong and you'll load a BatchNorm mean
into a conv kernel, silently — no shape error if sizes coincidentally align.

TF.js model.weights is NOT a naive per-layer concatenation. It is:

    [all trainable weights, in layer topo order] +
    [all non-trainable weights (BN running stats), in layer creation order]

So for BatchNorm, γ and β appear early (among trainable) while μ and σ² get
tacked on at the end. Additionally, because the policy and value heads both
branch off the shared residual tower, TF.js's topological sort interleaves
them — the observed order has value_conv + value_bn placed *before*
policy_conv in the trainable section, with value_fc1/fc2 after.

Concrete order (verified against a freshly-built TS ChessNet):

    TRAINABLE
      init_conv W
      init_bn γ, β
      for each res block:
          conv1 W, bn1 γ β, conv2 W, bn2 γ β,
          SE fc1 W b, SE fc2 W b
      value_conv W
      value_bn γ, β
      policy_conv W, b
      value_fc1 W, b
      value_fc2 (WDL) W, b

    NON-TRAINABLE (appended last, BNs in creation order)
      init_bn μ, σ²
      for each res block:  bn1 μ σ², bn2 μ σ²
      value_bn μ, σ²

Layout conversions handled on export:
  * Conv weights: PyTorch [outC, inC, kH, kW] → TF.js [kH, kW, inC, outC].
  * Linear weights: PyTorch [outF, inF] → TF.js [inF, outF].
  * BatchNorm scalars: same shape and per-variable identity in both frameworks.
"""

from __future__ import annotations

from typing import Any

import torch

from .model import ChessNet


def _conv_weight_ts(conv: torch.nn.Conv2d) -> torch.Tensor:
    # [outC, inC, kH, kW] → [kH, kW, inC, outC]
    return conv.weight.detach().permute(2, 3, 1, 0).contiguous()


def _linear_weight_ts(linear: torch.nn.Linear) -> torch.Tensor:
    # [outF, inF] → [inF, outF]
    return linear.weight.detach().t().contiguous()


def _append(shapes: list[list[int]], data: list[list[float]], tensor: torch.Tensor) -> None:
    shapes.append(list(tensor.shape))
    data.append(tensor.detach().cpu().float().contiguous().view(-1).tolist())


def export_weights(model: ChessNet, learning_rate: float = 1e-3) -> dict[str, Any]:
    """Serialize a PyTorch ChessNet into the browser's SerializedWeights JSON shape."""
    shapes: list[list[int]] = []
    data: list[list[float]] = []

    # ------------------------------------------------------------------
    # TRAINABLE WEIGHTS (γ, β but NOT μ/σ² — those go at the end)
    # ------------------------------------------------------------------

    # Initial conv + BN (γ, β)
    _append(shapes, data, _conv_weight_ts(model.init_conv))
    _append(shapes, data, model.init_bn.weight)
    _append(shapes, data, model.init_bn.bias)

    # Residual blocks
    for rb in model.res_blocks:
        _append(shapes, data, _conv_weight_ts(rb.conv1))
        _append(shapes, data, rb.bn1.weight)
        _append(shapes, data, rb.bn1.bias)

        _append(shapes, data, _conv_weight_ts(rb.conv2))
        _append(shapes, data, rb.bn2.weight)
        _append(shapes, data, rb.bn2.bias)

        _append(shapes, data, _linear_weight_ts(rb.se.fc1))
        _append(shapes, data, rb.se.fc1.bias)
        _append(shapes, data, _linear_weight_ts(rb.se.fc2))
        _append(shapes, data, rb.se.fc2.bias)

    # Heads — TF.js topo order interleaves value/policy. Observed order:
    #   value_conv W, value_bn γ β, policy_conv W b, value_fc1 W b, value_fc2 W b
    _append(shapes, data, _conv_weight_ts(model.value_conv))
    _append(shapes, data, model.value_bn.weight)
    _append(shapes, data, model.value_bn.bias)

    _append(shapes, data, _conv_weight_ts(model.policy_conv))
    _append(shapes, data, model.policy_conv.bias)

    _append(shapes, data, _linear_weight_ts(model.value_fc1))
    _append(shapes, data, model.value_fc1.bias)
    _append(shapes, data, _linear_weight_ts(model.value_fc2))
    _append(shapes, data, model.value_fc2.bias)

    # ------------------------------------------------------------------
    # NON-TRAINABLE WEIGHTS (all BN running stats, in BN creation order)
    # ------------------------------------------------------------------

    _append(shapes, data, model.init_bn.running_mean)
    _append(shapes, data, model.init_bn.running_var)

    for rb in model.res_blocks:
        _append(shapes, data, rb.bn1.running_mean)
        _append(shapes, data, rb.bn1.running_var)
        _append(shapes, data, rb.bn2.running_mean)
        _append(shapes, data, rb.bn2.running_var)

    _append(shapes, data, model.value_bn.running_mean)
    _append(shapes, data, model.value_bn.running_var)

    return {
        "config": {
            "numResBlocks": model.num_res_blocks,
            "numFilters": model.num_filters,
            "kernelSize": model.kernel_size,
            "valueHeadSize": model.value_head_size,
            "seReduction": model.se_reduction,
            "learningRate": learning_rate,
        },
        "shapes": shapes,
        "data": data,
    }


def _conv_weight_from_ts(flat: list[float], shape: list[int]) -> torch.Tensor:
    # TF.js shape [kH, kW, inC, outC] → PyTorch [outC, inC, kH, kW]
    t = torch.tensor(flat, dtype=torch.float32).view(*shape)
    return t.permute(3, 2, 0, 1).contiguous()


def _linear_weight_from_ts(flat: list[float], shape: list[int]) -> torch.Tensor:
    # TF.js [inF, outF] → PyTorch [outF, inF]
    t = torch.tensor(flat, dtype=torch.float32).view(*shape)
    return t.t().contiguous()


def import_weights(model: ChessNet, serialized: dict[str, Any]) -> None:
    """Load TS SerializedWeights back into a PyTorch ChessNet (inverse of export_weights)."""
    shapes = serialized["shapes"]
    data = serialized["data"]
    idx = 0

    def next_1d() -> torch.Tensor:
        nonlocal idx
        shape = shapes[idx]
        flat = data[idx]
        idx += 1
        return torch.tensor(flat, dtype=torch.float32).view(*shape)

    def next_conv() -> torch.Tensor:
        nonlocal idx
        shape = shapes[idx]
        flat = data[idx]
        idx += 1
        return _conv_weight_from_ts(flat, shape)

    def next_linear() -> torch.Tensor:
        nonlocal idx
        shape = shapes[idx]
        flat = data[idx]
        idx += 1
        return _linear_weight_from_ts(flat, shape)

    with torch.no_grad():
        # Trainable
        model.init_conv.weight.copy_(next_conv())
        model.init_bn.weight.copy_(next_1d())
        model.init_bn.bias.copy_(next_1d())

        for rb in model.res_blocks:
            rb.conv1.weight.copy_(next_conv())
            rb.bn1.weight.copy_(next_1d())
            rb.bn1.bias.copy_(next_1d())
            rb.conv2.weight.copy_(next_conv())
            rb.bn2.weight.copy_(next_1d())
            rb.bn2.bias.copy_(next_1d())
            rb.se.fc1.weight.copy_(next_linear())
            rb.se.fc1.bias.copy_(next_1d())
            rb.se.fc2.weight.copy_(next_linear())
            rb.se.fc2.bias.copy_(next_1d())

        # Heads (TF.js interleaved order)
        model.value_conv.weight.copy_(next_conv())
        model.value_bn.weight.copy_(next_1d())
        model.value_bn.bias.copy_(next_1d())
        model.policy_conv.weight.copy_(next_conv())
        model.policy_conv.bias.copy_(next_1d())
        model.value_fc1.weight.copy_(next_linear())
        model.value_fc1.bias.copy_(next_1d())
        model.value_fc2.weight.copy_(next_linear())
        model.value_fc2.bias.copy_(next_1d())

        # Non-trainable (BN running stats, in creation order)
        model.init_bn.running_mean.copy_(next_1d())
        model.init_bn.running_var.copy_(next_1d())
        for rb in model.res_blocks:
            rb.bn1.running_mean.copy_(next_1d())
            rb.bn1.running_var.copy_(next_1d())
            rb.bn2.running_mean.copy_(next_1d())
            rb.bn2.running_var.copy_(next_1d())
        model.value_bn.running_mean.copy_(next_1d())
        model.value_bn.running_var.copy_(next_1d())

    assert idx == len(shapes), f"Weight count mismatch: consumed {idx}, fixture has {len(shapes)}"
