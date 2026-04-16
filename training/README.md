# chess-ai training rig

Python/PyTorch training pipeline for the `ai-chess` browser app.

The browser app owns inference and UI. This directory owns *training*:

- a faithful Python port of the browser's `ChessEngine.ts` and encoder
- PyTorch ResNet (with SE blocks + WDL head) mirroring `ChessNet.ts`
- batched MCTS + self-play engine
- replay buffer + gradient-update training loop
- a weight exporter that emits `SerializedWeights` JSON the browser loads as-is

## Layout

```
training/
├── pyproject.toml
├── src/chess_ai/
│   ├── engine.py        # Board, Move, legal-move generation, apply_move
│   ├── encoding.py      # encode_board (20 planes), move_to_index
│   ├── rewards.py       # evaluate_position (multi-signal shaped reward)
│   ├── model.py         # PyTorch ChessNet (ResNet + SE + WDL)
│   ├── weight_io.py     # export_weights / import_weights (TS JSON format)
│   ├── mcts.py          # MCTS + batched multi-game search
│   ├── selfplay.py      # ReplayBuffer, SelfPlayEngine
│   └── train.py         # Main training loop + checkpoints
├── scripts/
│   ├── train.py                     # CLI entrypoint for training
│   ├── make_roundtrip_fixture.py    # Generates PyTorch round-trip fixture
│   └── deploy_to_browser.py         # Copies weights JSON into the browser preset
└── tests/
    ├── test_parity.py               # TS vs Python engine (legal moves, encoding, move idx)
    ├── test_reward_parity.py        # TS vs Python shaped reward
    ├── test_weight_roundtrip.py     # PyTorch ↔ TF.js forward-pass parity
    ├── test_selfplay_smoke.py       # Self-play pipeline runs cleanly
    └── test_train_smoke.py          # Full train loop + checkpoint round-trip
```

## Install

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
# Then install PyTorch for your accelerator (CUDA / MPS / CPU):
# https://pytorch.org/get-started/locally/
```

## Workflow: from scratch → browser-playable model

```bash
# 1. Generate parity fixture (from project root, Node side)
npm run dump-parity 5000

# 2. Verify Python ports match TS engine byte-for-byte
cd training && pytest

# 3. Train (on your CUDA box)
python scripts/train.py \
    --num-res-blocks 10 --num-filters 128 \
    --concurrent-games 64 --mcts-sims 50 \
    --batch-size 256 --replay-buffer 100000 \
    --lr 1e-3 \
    --device cuda \
    --checkpoint-dir runs/v1

# 4. Ship the weights to the browser preset slot
python scripts/deploy_to_browser.py runs/v1/latest.json

# 5. Rebuild the browser app (project root)
cd .. && npm run build
```

## Parity gates

Every time you change a chess rule, reward weight, or model layer you must
re-run the tests. These are ordered from cheap to expensive:

| test | what it checks |
|---|---|
| `test_parity` | Python engine = TS engine (legal moves, 20-plane encoding, policy index) |
| `test_reward_parity` | Python `evaluate_position` = TS `evaluatePosition` on same positions |
| `test_weight_roundtrip` | PyTorch forward = TF.js forward to 1e-4 on same weights |
| `test_selfplay_smoke` | Self-play engine produces examples without crashing |
| `test_train_smoke` | Full loop runs, checkpoints load back, JSON round-trips |

## Where to edit what

- **Chess rules / move generation** → `src/chess_ai/engine.py` + mirror in
  `src/game/chess/ChessEngine.ts`, then re-run `npm run dump-parity` and
  `pytest`.
- **Reward shaping** → `src/chess_ai/rewards.py` + `src/game/ai/rewardShaping.ts`.
- **Model architecture** → `src/chess_ai/model.py` + `src/game/ai/ChessNet.ts`
  + `src/game/ai/CPUForward.ts`. Also update the weight-order documentation
  at the top of `weight_io.py` if weight ordering shifts.
- **Training hyperparameters** → CLI flags in `scripts/train.py`.
- **Self-play scheduling** → `src/chess_ai/selfplay.py` (concurrent games,
  MCTS sim count, random-start distribution).
