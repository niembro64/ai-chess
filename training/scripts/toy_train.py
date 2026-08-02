"""Toy: the smallest chess net that actually trains — end to end in one file.

This is the pedagogical twin of the production pipeline ("Sage"). Same
skeleton — shared conv trunk, policy head says WHAT to do, value head
says HOW GOOD it is, both gradients shape the same trunk — with every
inessential removed so the whole thing fits in your head:

  input   8x8x6  one channel per piece TYPE; +1 = mover's piece,
                 -1 = opponent's, 0 = empty. Board is rotated 180°
                 when black is to move (mover always plays "up").
                 Deliberately blind to castling rights / en passant /
                 move clocks — watch the policy grid put mass on moves
                 the legal mask then erases.
  trunk   3 residual blocks x 32 filters, no batch-norm (keeps the
                 browser port and the visualization dead simple).
  policy  1x1 conv -> 4 ch -> flatten -> FC -> 4096 logits = the
                 64x64 from->to grid (promotions collapse to queen,
                 same as Sage).
  value   1x1 conv -> 2 ch -> flatten -> FC 64 -> FC 1 -> tanh,
                 mover's win probability-ish in [-1, 1].
  loss    cross-entropy(policy, MCTS visit distribution)
          + MSE(value, final outcome) + weight decay (AdamW).

Self-play is intentionally the simple version: one game at a time,
batch-1 NN calls, ~50-sim MCTS. Readability is the product; expect a
few hundred games per hour on an Apple-Silicon Mac, which takes the
net from random flailing to "develops pieces and grabs hanging
material" overnight.

Usage:
    python scripts/toy_train.py                    # train (resumes toy-latest.pt)
    python scripts/toy_train.py --iterations 50
    python scripts/toy_train.py --dump-fixture f.json   # parity fixture for TS

Checkpoints land in training/toy_checkpoints/: toy-latest.pt (resume),
toy-latest.json (browser format), toy-iter-N.json snapshots. Deploy to
the site with:  cp training/toy_checkpoints/toy-latest.json public/models/toy.json
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_ai.encoding import move_to_index  # noqa: E402  (same 64x64 indexing as Sage)
from chess_ai.engine import (  # noqa: E402
    ChessGameState,
    apply_move,
    create_initial_game_state,
    expand_children,
    get_legal_moves,
    is_in_check,
)
from chess_ai.selfplay import _is_insufficient_material, _position_key  # noqa: E402

CKPT_DIR = ROOT / "toy_checkpoints"

# --- The 6-plane encoding ---------------------------------------------

NUM_PLANES = 6
PIECE_CHANNEL = {"pawn": 0, "knight": 1, "bishop": 2, "rook": 3, "queen": 4, "king": 5}
POLICY_SIZE = 4096


def encode_toy(state: ChessGameState) -> np.ndarray:
    """Flat [8*8*6] float32, channels-last: idx = (rank*8 + file)*6 + ch.

    Mover's perspective: own pieces +1, opponent -1; board rotated 180°
    (rank AND file) when black moves — identical convention to Sage, so
    positions/moves stay mentally comparable between the two nets.
    """
    x = np.zeros(8 * 8 * NUM_PLANES, dtype=np.float32)
    white_to_move = state.currentTurn == "white"
    for r in range(8):
        for f in range(8):
            p = state.board[r][f]
            if p is None:
                continue
            rr, ff = (r, f) if white_to_move else (7 - r, 7 - f)
            sign = 1.0 if (p.color == "white") == white_to_move else -1.0
            x[(rr * 8 + ff) * NUM_PLANES + PIECE_CHANNEL[p.type]] = sign
    return x


# --- The network -------------------------------------------------------

TOY_FILTERS = 32
TOY_BLOCKS = 3
POLICY_CH = 4
VALUE_CH = 2
VALUE_HIDDEN = 64


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(NUM_PLANES, TOY_FILTERS, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(TOY_BLOCKS):
            self.blocks.append(nn.ModuleDict({
                "conv1": nn.Conv2d(TOY_FILTERS, TOY_FILTERS, 3, padding=1),
                "conv2": nn.Conv2d(TOY_FILTERS, TOY_FILTERS, 3, padding=1),
            }))
        self.policy_conv = nn.Conv2d(TOY_FILTERS, POLICY_CH, 1)
        self.policy_fc = nn.Linear(8 * 8 * POLICY_CH, POLICY_SIZE)
        self.value_conv = nn.Conv2d(TOY_FILTERS, VALUE_CH, 1)
        self.value_fc1 = nn.Linear(8 * 8 * VALUE_CH, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    @staticmethod
    def _flatten_hwc(t: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC -> flat, so the browser (channels-last) flattens
        # in exactly the same order. THE load-bearing transpose.
        return t.permute(0, 2, 3, 1).reshape(t.shape[0], -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv_in(x))
        for b in self.blocks:
            y = b["conv2"](F.relu(b["conv1"](h)))
            h = F.relu(h + y)
        p = self._flatten_hwc(F.relu(self.policy_conv(h)))
        policy_logits = self.policy_fc(p)
        v = self._flatten_hwc(F.relu(self.value_conv(h)))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return policy_logits, value.squeeze(-1)


def flat_to_nchw(boards: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(boards.reshape(-1, 8, 8, NUM_PLANES))
    return t.permute(0, 3, 1, 2).contiguous()


# --- Browser weight export ---------------------------------------------
# Fixed tensor order; conv filters transposed to TF layout [h,w,in,out],
# linear weights to [in,out]; fp16 base64 like Sage's format.

def export_toy_json(model: ToyNet) -> dict:
    tensors: list[tuple[str, np.ndarray]] = []

    def conv(name: str, m: nn.Conv2d) -> None:
        tensors.append((f"{name}.w", m.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)))
        tensors.append((f"{name}.b", m.bias.detach().cpu().numpy()))

    def fc(name: str, m: nn.Linear) -> None:
        tensors.append((f"{name}.w", m.weight.detach().cpu().numpy().T))
        tensors.append((f"{name}.b", m.bias.detach().cpu().numpy()))

    conv("conv_in", model.conv_in)
    for i, b in enumerate(model.blocks):
        conv(f"block{i}.conv1", b["conv1"])
        conv(f"block{i}.conv2", b["conv2"])
    conv("policy_conv", model.policy_conv)
    fc("policy_fc", model.policy_fc)
    conv("value_conv", model.value_conv)
    fc("value_fc1", model.value_fc1)
    fc("value_fc2", model.value_fc2)

    return {
        "kind": "toy-v1",
        "config": {
            "numPlanes": NUM_PLANES,
            "numFilters": TOY_FILTERS,
            "numResBlocks": TOY_BLOCKS,
            "policyChannels": POLICY_CH,
            "valueChannels": VALUE_CH,
            "valueHidden": VALUE_HIDDEN,
        },
        "names": [n for n, _ in tensors],
        "shapes": [list(a.shape) for _, a in tensors],
        "data": [
            base64.b64encode(a.astype("<f2").tobytes()).decode("ascii")
            for _, a in tensors
        ],
    }


# --- Minimal MCTS (the whole algorithm, no tricks) ----------------------

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25


class Node:
    __slots__ = ("state", "parent", "move", "children", "visits", "total_value",
                 "prior", "terminal_value", "is_terminal", "expanded")

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: dict[int, Node] = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.expanded = False
        s = state.status
        self.is_terminal = s in ("checkmate", "stalemate", "draw")
        # From the perspective of the side to move AT this node: being
        # checkmated is -1; stalemate / 50-move draw is 0.
        self.terminal_value = -1.0 if s == "checkmate" else 0.0


def _expand(node: Node, policy: np.ndarray) -> None:
    children = expand_children(node.state)
    if not children:
        node.is_terminal = True
        node.terminal_value = -1.0 if is_in_check(node.state.board, node.state.currentTurn) else 0.0
        return
    is_white = node.state.currentTurn == "white"
    total = 0.0
    entries = []
    for move, child_state in children:
        mi = move_to_index(move, is_white)
        if mi in node.children:
            continue  # underpromotions collapse onto the queen slot
        node.children[mi] = Node(child_state, parent=node, move=move)
        entries.append(mi)
        total += float(policy[mi])
    for mi in entries:
        node.children[mi].prior = float(policy[mi]) / total if total > 0 else 1.0 / len(entries)
    node.expanded = True


def _backprop(node: Node, value: float) -> None:
    v = value
    while node is not None:
        node.visits += 1
        node.total_value += v
        v = -v
        node = node.parent


def mcts(state, net_eval, sims: int, rng: random.Random, root_noise: bool):
    """Run `sims` simulations; return (visit_policy[4096], root)."""
    root = Node(state)
    policy, value = net_eval(encode_toy(state))
    _expand(root, policy)
    if root_noise and root.children:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
        for i, c in enumerate(root.children.values()):
            c.prior = (1 - DIRICHLET_EPS) * c.prior + DIRICHLET_EPS * float(noise[i])
    _backprop(root, value)

    for _ in range(sims):
        node = root
        # Select: descend by PUCT. A child's stored value is from the
        # CHILD's perspective, so negate when viewing from the parent.
        while node.expanded and not node.is_terminal:
            sqrt_n = math.sqrt(max(1, node.visits))
            best, best_score = None, -1e9
            for c in node.children.values():
                q = -c.total_value / c.visits if c.visits > 0 else 0.0
                u = C_PUCT * c.prior * sqrt_n / (1 + c.visits)
                if q + u > best_score:
                    best, best_score = c, q + u
            node = best
        if node.is_terminal:
            _backprop(node, node.terminal_value)
            continue
        policy, value = net_eval(encode_toy(node.state))
        _expand(node, policy)
        _backprop(node, value)

    visit_policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    total = sum(c.visits for c in root.children.values())
    if total > 0:
        for mi, c in root.children.items():
            visit_policy[mi] = c.visits / total
    return visit_policy, root


# --- Self-play ----------------------------------------------------------

MOVE_CAP = 160
TEMP_PLIES = 20


def play_game(net_eval, sims: int, rng: random.Random):
    """One self-play game. Returns (examples, outcome_str) where each
    example is (planes, visit_policy, mover_color)."""
    state = create_initial_game_state()
    state.status = "active"
    counts = {_position_key(state): 1}
    examples = []
    outcome = None  # +1 white wins, -1 black wins, 0 draw
    label = "cap"

    for ply in range(MOVE_CAP):
        visit_policy, root = mcts(state, net_eval, sims, rng, root_noise=True)
        examples.append((encode_toy(state), visit_policy, state.currentTurn))

        children = list(root.children.values())
        if not children:
            break
        if ply < TEMP_PLIES:
            weights = [c.visits for c in children]
            move = rng.choices(children, weights=weights)[0].move
        else:
            move = max(children, key=lambda c: c.visits).move

        state = apply_move(state, move)

        if state.status == "checkmate":
            outcome = 1.0 if state.currentTurn == "black" else -1.0
            label = "mate"
            break
        if state.status in ("stalemate", "draw"):
            outcome, label = 0.0, state.status
            break
        key = _position_key(state)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 3:
            outcome, label = 0.0, "repetition"
            break
        if _is_insufficient_material(state.board):
            outcome, label = 0.0, "insufficient"
            break

    if outcome is None:
        outcome, label = 0.0, "cap"

    out = []
    for planes, pol, mover in examples:
        v = outcome if mover == "white" else -outcome
        out.append((planes, pol, np.float32(v)))
    return out, label


# --- Training loop -------------------------------------------------------

def make_net_eval(model: ToyNet, device: torch.device):
    def net_eval(planes: np.ndarray):
        with torch.no_grad():
            x = flat_to_nchw(planes[None, :]).to(device)
            logits, value = model(x)
            policy = F.softmax(logits, dim=-1)[0].cpu().numpy()
        return policy, float(value[0])
    return net_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Toy net.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iter", type=int, default=6)
    parser.add_argument("--sims", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer", type=int, default=30_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Plain per-iteration lines instead of the live "
                             "dashboard (auto-disabled when stdout is not a TTY)")
    parser.add_argument("--dump-fixture", type=Path, default=None,
                        help="Write a TS-parity fixture (positions + outputs) and exit")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ToyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    start_iter = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    latest = CKPT_DIR / "toy-latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck.get("iteration", 0)
        print(f"resumed from {latest} at iteration {start_iter}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ToyNet: {n_params/1e6:.2f}M params on {device}")

    if args.dump_fixture is not None:
        _dump_fixture(model, device, args.dump_fixture, rng)
        return

    buffer: deque = deque(maxlen=args.buffer)

    # Plain lines always go to the log file (so remote tails and tools
    # can read progress regardless of the dashboard) and to CSV for
    # later plotting. The rich dashboard renders on top when we have a
    # real terminal.
    log_path = CKPT_DIR / "train.log"
    csv_path = CKPT_DIR / "toy_stats.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "time,iteration,games,buffer,p_loss,v_loss,selfplay_s,train_s,labels\n"
        )

    use_dashboard = sys.stdout.isatty() and not args.no_dashboard
    dashboard = None
    if use_dashboard:
        try:
            from toy_dashboard import ToyDashboard
            dashboard = ToyDashboard(args.iterations, str(device), n_params)
        except Exception as e:
            print(f"dashboard unavailable ({e}); falling back to plain output")

    dash_ctx = dashboard if dashboard is not None else contextlib.nullcontext()
    with dash_ctx:
        _train_loop(args, model, optimizer, buffer, start_iter, device,
                    dashboard, log_path, csv_path)


def _train_loop(args, model, optimizer, buffer, start_iter, device,
                dashboard, log_path: Path, csv_path: Path) -> None:
    rng = random.Random(args.seed + start_iter)

    for it in range(start_iter, args.iterations):
        model.eval()
        net_eval = make_net_eval(model, device)
        t0 = time.time()
        labels: dict[str, int] = {}
        n_examples = 0
        for _ in range(args.games_per_iter):
            examples, label = play_game(net_eval, args.sims, rng)
            labels[label] = labels.get(label, 0) + 1
            buffer.extend(examples)
            n_examples += len(examples)
        selfplay_s = time.time() - t0

        model.train()
        n_steps = max(1, n_examples // 64)
        t1 = time.time()
        p_loss = v_loss = 0.0
        for _ in range(n_steps):
            batch = rng.sample(range(len(buffer)), min(args.batch_size, len(buffer)))
            boards = np.stack([buffer[i][0] for i in batch])
            policies = np.stack([buffer[i][1] for i in batch])
            values = np.stack([buffer[i][2] for i in batch])

            x = flat_to_nchw(boards).to(device)
            pt = torch.from_numpy(policies).to(device)
            vt = torch.from_numpy(values).to(device)

            logits, v = model(x)
            policy_loss = -(pt * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
            value_loss = F.mse_loss(v, vt)
            loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            p_loss, v_loss = policy_loss.item(), value_loss.item()
        train_s = time.time() - t1

        line = (
            f"iter {it + 1:4d}  games {args.games_per_iter} ({labels})  "
            f"buffer {len(buffer):6d}  p_loss {p_loss:.3f}  v_loss {v_loss:.3f}  "
            f"selfplay {selfplay_s:5.1f}s  train {train_s:4.1f}s"
        )
        with log_path.open("a") as f:
            f.write(line + "\n")
        with csv_path.open("a") as f:
            labels_str = ";".join(f"{k}:{n}" for k, n in sorted(labels.items()))
            f.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{it + 1},"
                f"{args.games_per_iter},{len(buffer)},{p_loss:.4f},{v_loss:.4f},"
                f"{selfplay_s:.1f},{train_s:.2f},{labels_str}\n"
            )
        if dashboard is not None:
            dashboard.on_iteration(
                it + 1, labels, len(buffer), p_loss, v_loss, selfplay_s, train_s,
            )
        else:
            print(line, flush=True)

        latest = CKPT_DIR / "toy-latest.pt"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "iteration": it + 1},
            latest,
        )
        with (CKPT_DIR / "toy-latest.json").open("w") as f:
            json.dump(export_toy_json(model), f)
        if (it + 1) % 25 == 0:
            with (CKPT_DIR / f"toy-iter-{it + 1}.json").open("w") as f:
                json.dump(export_toy_json(model), f)


def _dump_fixture(model: ToyNet, device: torch.device, path: Path,
                  rng: random.Random) -> None:
    """Positions + this net's outputs, for the TS parity check
    (scripts/verify_toy_weights.ts)."""
    model.eval()
    states = []
    s = create_initial_game_state()
    s.status = "active"
    states.append(s)
    for _ in range(7):
        moves = get_legal_moves(s)
        if not moves:
            break
        s = apply_move(s, rng.choice(moves))
        if s.status in ("checkmate", "stalemate", "draw"):
            break
        states.append(s)

    boards = np.stack([encode_toy(st) for st in states])
    with torch.no_grad():
        logits, values = model(flat_to_nchw(boards).to(device))
        policies = F.softmax(logits, dim=-1).cpu().numpy()
    fixture = {
        "boards": [b.tolist() for b in boards],
        "policies": [p.tolist() for p in policies],
        "values": [float(v) for v in values.cpu()],
    }
    with path.open("w") as f:
        json.dump(fixture, f)
    print(f"fixture with {len(states)} positions -> {path}")


if __name__ == "__main__":
    main()
