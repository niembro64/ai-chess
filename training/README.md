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
│   ├── _launcher.py                 # Shared config-driven training launcher
│   ├── train_ubuntu_new.py          # Entrypoint: 4-core + RTX 3090, fresh
│   ├── train_ubuntu_continue.py     #             same, warm-start latest.pt
│   ├── train_windows_new.py         # Entrypoint: 9900K + 1080 Ti, fresh
│   ├── train_windows_continue.py    #             same, warm-start latest.pt
│   ├── train_mac_new.py             # Entrypoint: Apple Silicon (MPS), fresh
│   ├── train_mac_continue.py        #             same, warm-start latest.pt
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
pip install maturin
# Then install PyTorch for your accelerator (CUDA / MPS / CPU):
# https://pytorch.org/get-started/locally/
```

**Build the Rust engine extension** (required for fast self-play — ~10× speedup
on the MCTS hot path). Needs a Rust toolchain (`curl https://sh.rustup.rs -sSf | sh`):

```bash
cd rust_engine
maturin develop --release
cd ..
```

`get_legal_moves` auto-detects the Rust module and dispatches to it. If the
build isn't done, Python falls back to the pure-Python engine silently (much
slower, but fully functional).

## Workflow: from scratch → browser-playable model

```bash
# 1. Generate parity fixture (from project root, Node side)
npm run dump-parity 5000

# 2. Verify Python ports match TS engine byte-for-byte
cd training && pytest

# 3. Train (on your CUDA box). Run inside tmux so the session survives
#    disconnections; the dashboard repaints the moment you re-attach.
tmux new -s train

# Checkpoints land in runs/latest/ (single fixed spot — re-running
# overwrites the previous checkpoint; there is no v1/v2 scheme).
# Everything (MCTS depth, lr schedule, eval cadence, arch) lives in
# training/config.py. Pick the entrypoint matching your hardware AND
# whether you're starting fresh or continuing from latest.pt:
python scripts/train_ubuntu_new.py            # 4-core + RTX 3090, fresh
python scripts/train_ubuntu_continue.py       # same, resume runs/latest/latest.pt
python scripts/train_windows_new.py           # 9900K + 1080 Ti, fresh
python scripts/train_windows_continue.py      # same, resume
python scripts/train_mac_new.py               # Apple Silicon, fresh
python scripts/train_mac_continue.py          # same, resume
# Detach with Ctrl-b d; re-attach with `tmux a -t train`.
# `*_continue.py` errors out if runs/latest/latest.pt is missing —
# never silently bootstraps. Use --resume <path> on any entrypoint
# to override with an archived checkpoint.

# 4. Ship the weights to the browser preset slot
python scripts/deploy_to_browser.py runs/latest/latest.json

# 5. Rebuild the browser app (project root)
cd .. && npm run build
```

## Live training dashboard

When running in a terminal (TTY), the training launcher paints a Rich-based
TUI in the current tmux pane:

- `progress` — step, gen, games, games/min, replay size
- `model` — architecture summary + device + path to the CSV log
- `outcomes` — cumulative W/B/D counts with percent bars
- `loss (rolling)` — ASCII line plot of policy/value/total loss over the
  most recent 500 gradient steps (via `plotext`)
- `events` — last handful of log messages (checkpoints, etc.)

The same data streams into `<checkpoint_dir>/stats.csv` for offline analysis:

```python
import pandas as pd
df = pd.read_csv("runs/latest/stats.csv")
df.plot(x="step", y=["policy_loss", "value_loss"])
```

Set `ENABLE_DASHBOARD = False` in `config.py` (or redirect stdout to a file)
to fall back to plain text logging, which is friendlier for `nohup`/systemd-
style runs.

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
- **Training hyperparameters** → `training/config.py` (single source of truth).
- **Self-play scheduling** → `src/chess_ai/selfplay.py` (concurrent games,
  MCTS sim count, random-start distribution).
