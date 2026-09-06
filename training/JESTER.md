JESTER plays competitive inverted chess: the player whose own king is checkmated wins. Legal chess moves, stalemate, repetition, and insufficient-material rules still apply. Both sides try to force their own mate; neither is assumed to cooperate.

The policy/value architecture and exported weight formats stay compatible with SAGE and existing JESTER checkpoints. Values retain the ordinary-outcome sign convention (the mated side gets -1); search selection is inverted at both colors. Do not also reverse value targets. They describe outcomes under inverted play, not optimal ordinary chess.

`build_jester_config()` now uses 256 simulations, Rust trees, multiple workers, and a central GPU inference server. Half of new games initially start from independently verified short selfmates; the remainder use standard or legal random-walk starts. The curriculum contains 36 base positions with one-, two-, and three-move proofs, split before color augmentation into training and held-out sets. Its first motif family uses discovered rook mates; it is a bootstrap curriculum, not a comprehensive measure of inverted-chess strength. The solver checks all legal opponent replies and never labels budget exhaustion as a proof.

Current-network self-play receives 75% of games; 25% play frozen historical JESTERs when a pool is available. Both participants seek their own mate. Only learner plies from historical-opponent games enter replay. Cooperative mate acceptance, random-move overrides and SAGE opponents are disabled. Ordinary Syzygy is disabled in the launcher and explicitly bypassed in JESTER adjudication. Real draws receive full value weight; move-cap outcomes remain masked for value loss. Resignation and value-label distance decay stay disabled.

Promotion uses batched matches against the champion and distinct frozen opponents, paired colors, curated positions, and fresh openings from explicit seeds. No root noise or artificial mate acceptance is used. Caps are recorded separately from legal draws. The reported score gives caps half a point, but the approximate 95% promotion interval treats them pessimistically. Promotion requires score >= 0.55, lower bound > 0.5, and at least 50% held-out tactical accuracy. The curriculum share decreases only when all three held-out depths reach 80%, down to a 10% floor. Draw-heavy evaluations do not automatically stop training.

The dashboard and stats.csv report own-mate wins, delivered-mate losses, draws, and caps by opponent/start distribution. eval.csv includes uncertainty, caps, and tactical scores; eval_games.jsonl keeps individual match outcomes. The optional fumbler diagnostic is separate from competitive promotion. Its cache uses full position identity, seat, stable seed and search settings instead of reusable display names and Python hash().

To start a separate experiment from old weights on the training host:

```sh
cd /home/gpus/ai-chess/training
.venv/bin/pip install -e '.[dev]'
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release --manifest-path rust_engine/Cargo.toml
.venv/bin/python scripts/train_jester_new.py \
  --init-from runs_jester/latest/champion.pt \
  --opponent runs_jester/latest/champion.pt \
  --opponent runs_jester/latest/archive/gen-5480.pt \
  --checkpoint-dir runs_jester/competitive
```

`--init-from` copies only model weights. Replay, optimizer, generation count and evaluation history start fresh; initialization.json records the source and hash. Opponents are copied into the new run's opponents/ directory, so the experiment does not depend on mutable external files. The old runs_jester/latest directory remains intact. Starting fresh into an occupied output directory is rejected.

For later resumes use `scripts/train_jester_continue.py --checkpoint-dir runs_jester/competitive`. An old cooperative checkpoint must use `--init-from` in a new directory, not `--resume`. Workers start with the loaded weights, resume the curriculum setting, and are checked for crashes instead of leaving a silently starved trainer. Checkpoint replacement is atomic. `--workers`, `--sims`, `--steps`, and `--no-dashboard` support controlled smoke runs.

Read-only diagnostics:

```sh
.venv/bin/python scripts/benchmark_jester.py runs_jester/competitive/champion.pt --device cuda
.venv/bin/python scripts/benchmark_jester.py runs_jester/competitive/champion.pt --sims 256 --fumbler-games 10
.venv/bin/python scripts/build_selfmate_curriculum.py
.venv/bin/python -m pytest tests -q
```

On September 6, generation-7,416 exported weights solved 0/18 held-out starts at 96 simulations and 2/18 at both 256 and 400. Local MPS times were about 5, 7 and 10 seconds respectively. This small baseline motivates the curriculum and 256-simulation initial budget; it is not a competitive strength rating.

Rust and Python searches now share dual-net routing, inversion, terminal-distance preferences, root FPU and repetition semantics. Both deployed search callers use both-color inversion for JESTER. All four promotion choices remain searchable without expanding the existing 4096-output policy head: they share prior mass and aggregate training visits. Search chooses among promotions; raw-policy-only inference still cannot learn separate promotion logits. Natural JESTER follows actual game repetition rules, allowing legal return moves that the teaching-mode own-army veto would otherwise forbid. The alternate-goal ranking UI and existing weight assets remain unchanged.
