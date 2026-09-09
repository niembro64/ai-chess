# JESTER Uncheck Chess training

JESTER protocol 3 trains `uncheck-v1`. A player wins when their turn begins with their own king attacked. Kings may enter attack and are never captured. If any capture exists, the player must capture. Castling ignores attacked-square restrictions but is suppressed whenever a capture exists. Third repetition, 100 halfmoves, and no permitted moves draw; ordinary insufficient material does not.

The network architecture, 20 input planes, 4096 policy actions, WDL head, and serialized tensor order remain unchanged. A protocol-2 JESTER checkpoint may initialize protocol 3 with `--init-from`. This copies all parameters and BatchNorm buffers exactly while resetting the optimizer, replay, generation, curriculum, evaluation, and learning-rate state. `initialization.json` records both tensor digests and requires exact equality. Never use `--resume` across the protocol boundary.

Values use the explicit reference convention `uncheck-reference-v1`:

```text
z_actual(player) = +1 actual Uncheck win, -1 actual loss, 0 rule-defined draw
z_ref(player)    = -z_actual(player)
```

JESTER's existing both-color search inversion remains enabled. The target is negated exactly once; the evaluator, WDL channels, and search must not receive another sign swap. Dashboards and evaluations always report actual winners, even though replay stores reference values.

The initial production profile uses the existing 12-block, 160-filter model, three workers with 32 concurrent games each, 256 self-play simulations, batch size 256, learning rate `3e-4`, and a 200-ply truncation cap. Twenty-five percent of starts come from verified Uncheck proofs. Seventy-five percent are competitive standard or Uncheck-reachable openings. Competitive games use the current network 75% of the time and frozen JESTER history 25% of the time. Caps remain unknown and are masked for value training.

The curriculum uses variant-native move generation and bounded adversarial proofs. Training and held-out sets have distinct material signatures, so no board reflection or color reversal can cross the split. The catalog covers knight, rook, bishop, pawn and double attacks, discovered exposure, en passant, castling, compulsory remote captures, and all four capture promotions. The older selfmate catalog stays available only for protocol-2 regression tests.

Protocol 3 disables helper opponents, the old bridge pool, SAGE opponents, random mate acceptance, material auxiliary loss, automated resignation, and Syzygy labels. New bridge positions may be added only from validated Uncheck games.

Competitive evaluation batches current JESTER against the current champion and distinct frozen historical JESTERs under `uncheck-v1`. Every fixed standard or held-out opening is played with the candidate as each color. Held-out tactical recognition and full conversion remain diagnostics and do not add easy games to the promotion score. `eval.csv` and `eval_games.jsonl` retain actual W/D/L, unresolved caps, average plies, color split, start type, rescue attempts and successes, exposure conversions, and immediate opponent-win errors. Promotion requires score at least 55% and a paired conservative 95% lower bound above 50%; caps receive zero in that lower bound.

Start from preserved protocol-2 weights in a new directory:

```sh
cd /home/gpus/ai-chess-uncheck-<commit>/training
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CHESS_AI_NO_COMPILE=1 \
.venv/bin/python scripts/train_jester_new.py \
  --ruleset uncheck-v1 \
  --init-from /absolute/path/to/jester-source.pt \
  --checkpoint-dir /absolute/path/to/runs_jester/uncheck-v1-<timestamp> \
  --workers 3 --sims 256
```

Resume that same run only after it has a protocol-3 checkpoint:

```sh
.venv/bin/python scripts/train_jester_continue.py \
  --ruleset uncheck-v1 \
  --resume /absolute/path/to/uncheck-run/latest.pt \
  --checkpoint-dir /absolute/path/to/uncheck-run \
  --workers 3 --sims 256
```

A valid cutover preserves the complete protocol-2 run, archives its final checkpoint and checksums, uses a detached checkout of a tested commit and a separate CUDA environment, rebuilds the Rust extension with engine protocol 3, and runs the complete regression suite. A short training proof must show exact source tensor import, fresh replay and optimizer state, finite losses, changed finite parameters, and actual `uncheck_w` or `uncheck_b` outcomes before the persistent `uncheck` tmux session starts.
