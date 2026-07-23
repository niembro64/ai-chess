//! MCTS in Rust — arena-allocated tree, PUCT selection, terminal backup.
//!
//! Intended to replace the per-sim Python loop in `chess_ai.mcts.MCTSSearch`.
//! Python retains the outer batching loop (`run_batched_mcts`) because the
//! NN evaluator is Python/PyTorch and the per-sim FFI boundary is minor
//! compared to the dozens of PUCT + backprop ops each sim does.
//!
//! API (exposed to Python as `chess_ai_rust.MctsSearch`):
//!   - `new(state_dict, c_puct)`
//!   - `is_terminal()`
//!   - `get_root_board() -> bytes`   (NN-ready f32 tensor, same layout as
//!                                    chess_ai.encoding.encode_board)
//!   - `init_root(priors: bytes, value: float,
//!                noise_epsilon: float, noise_alpha: float, seed: int)`
//!   - `select_leaf() -> bytes | None`
//!   - `supply_eval(priors: bytes, value: float)`
//!   - `get_result(temperature: float, seed: int)
//!        -> (policy: bytes, move_tuple, root_value: float)`
//!
//! Parity semantics:
//! - PUCT score: `q + c_puct * prior * sqrt(parent.visits) / (1 + child.visits)`
//!   where `q = -child.total_value / child.visit_count` (parent-perspective).
//! - Backprop flips sign each ply (zero-sum convention).
//! - Terminal value: -1.0 for checkmate, 0.0 for stalemate/draw (from the
//!   losing side-to-move's perspective — mirrors Python).
//! - Dirichlet(alpha) noise at root: `prior = (1-eps)*prior + eps*noise`.
//! - Underpromotions share a policy index with queen promotions (matches
//!   the Python `move_to_index` collision on purpose — TS parity).

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};

use crate::{
    apply_move_full_core, is_in_check, legal_moves_impl, pack_board, pack_castling,
    pack_en_passant, piece_type_to_str, plane_index, Board, CastlingRights, Move,
    NUM_PLANES, PAWN,
};

const POLICY_SIZE: usize = 4096;
const BOARD_BYTES: usize = 8 * 8 * NUM_PLANES * 4; // f32 row-major

// Sentinel for "no parent" without Option overhead on the hot path.
const NO_PARENT: u32 = u32::MAX;

// ---- state stored per node (tree arena) ----

#[derive(Clone)]
struct NodeState {
    board: Board,
    castling: CastlingRights,
    ep: Option<(u8, u8)>,
    hmc: i32,
    fmn: i32,
    white_to_move: bool,
    status: NodeStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeStatus {
    Active,
    Check,
    Checkmate,
    Stalemate,
    Draw,
}

impl NodeStatus {
    fn from_str(s: &str) -> Self {
        match s {
            "checkmate" => NodeStatus::Checkmate,
            "stalemate" => NodeStatus::Stalemate,
            "draw" => NodeStatus::Draw,
            "check" => NodeStatus::Check,
            _ => NodeStatus::Active,
        }
    }
    fn is_terminal(self) -> bool {
        matches!(
            self,
            NodeStatus::Checkmate | NodeStatus::Stalemate | NodeStatus::Draw
        )
    }
    fn terminal_value(self) -> f32 {
        match self {
            NodeStatus::Checkmate => -1.0,
            NodeStatus::Stalemate | NodeStatus::Draw => 0.0,
            _ => 0.0,
        }
    }
}

#[derive(Clone)]
struct MctsNode {
    parent: u32,
    children: Vec<(u16, u32)>, // (policy_idx, child_node_idx)
    move_info: MoveInfo,       // the move that led to this node (from parent)
    visit_count: u32,
    total_value: f64,
    prior: f32,
    is_expanded: bool,
    is_terminal: bool,
    terminal_value: f32,
}

#[derive(Clone, Copy, Default)]
struct MoveInfo {
    from_r: u8,
    from_f: u8,
    to_r: u8,
    to_f: u8,
    promotion: i8, // 0 or KING..PAWN code
}

// ---- PyClass ----

#[pyclass]
pub struct MctsSearch {
    nodes: Vec<MctsNode>,
    states: Vec<NodeState>,
    pending_leaf: Option<u32>,
    c_puct: f32,
    /// FPU (First-Play Urgency) reduction. Unvisited children get an
    /// initial Q of `parent_Q - fpu_reduction` (from parent's perspective)
    /// so exploration prefers proven moves over prior-peaked untried ones.
    /// 0.0 = unvisited children score as "neutral" (old behavior).
    /// Leela/AlphaZero-derived rigs use ~0.4–0.5.
    fpu_reduction: f32,
}

#[pymethods]
impl MctsSearch {
    /// Construct a search rooted at the supplied game state dict (mirrors
    /// `ChessGameState.to_dict()` from Python). `c_puct` defaults to 1.5,
    /// `fpu_reduction` defaults to 0.0 (back-compat with pre-FPU behavior).
    #[new]
    #[pyo3(signature = (state_dict, c_puct=None, fpu_reduction=None))]
    fn new(
        state_dict: &Bound<'_, PyDict>,
        c_puct: Option<f32>,
        fpu_reduction: Option<f32>,
    ) -> PyResult<Self> {
        let root_state = parse_state_dict(state_dict)?;
        let is_terminal = root_state.status.is_terminal();
        let terminal_value = root_state.status.terminal_value();
        let root_node = MctsNode {
            parent: NO_PARENT,
            children: Vec::new(),
            move_info: MoveInfo::default(),
            visit_count: 0,
            total_value: 0.0,
            prior: 0.0,
            is_expanded: is_terminal,
            is_terminal,
            terminal_value,
        };
        Ok(MctsSearch {
            nodes: vec![root_node],
            states: vec![root_state],
            pending_leaf: None,
            c_puct: c_puct.unwrap_or(1.5),
            fpu_reduction: fpu_reduction.unwrap_or(0.0),
        })
    }

    fn is_terminal(&self) -> bool {
        self.nodes[0].is_terminal
    }

    /// Q of the best VISITED root child, from the root player's
    /// perspective: max over children of -(total_value / visit_count).
    /// This is the resignation statistic — "even my best move loses" —
    /// unlike the root's mean Q, which is dragged down by the forced
    /// exploration of the mover's own bad moves and so systematically
    /// over-triggers resignation. Falls back to the root mean Q when no
    /// child has been visited yet.
    fn best_child_q(&self) -> f32 {
        let root = &self.nodes[0];
        let mut best: Option<f32> = None;
        for &(_, ci) in &root.children {
            let c = &self.nodes[ci as usize];
            if c.visit_count > 0 {
                let q = -(c.total_value as f32) / c.visit_count as f32;
                if best.map_or(true, |b| q > b) {
                    best = Some(q);
                }
            }
        }
        best.unwrap_or(if root.visit_count > 0 {
            (root.total_value / root.visit_count as f64) as f32
        } else {
            0.0
        })
    }

    /// NN-ready encoded board for the root position. Bit-exact with
    /// `chess_ai.encoding.encode_board`.
    fn get_root_board<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut buf = vec![0u8; BOARD_BYTES];
        write_encoded_board(&self.states[0], buf.as_mut_slice());
        Ok(PyBytes::new_bound(py, &buf))
    }

    /// Set root priors from the NN, add Dirichlet noise (epsilon/alpha/seed),
    /// and backprop the root value. Priors must be the full 4096 f32 policy
    /// as bytes.
    fn init_root(
        &mut self,
        priors: &[u8],
        value: f32,
        noise_epsilon: f32,
        noise_alpha: f32,
        seed: u64,
    ) -> PyResult<()> {
        if priors.len() != POLICY_SIZE * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "priors must be {} bytes, got {}",
                POLICY_SIZE * 4,
                priors.len()
            )));
        }
        let priors_f32 = bytes_to_f32(priors);
        self.expand_with_policy(0, priors_f32);
        if noise_epsilon > 0.0 && noise_alpha > 0.0 {
            self.add_dirichlet_noise(0, noise_epsilon, noise_alpha, seed);
        }
        self.backpropagate(0, value);
        Ok(())
    }

    /// Descend root → expanded leaf; if the leaf is terminal, backprop
    /// immediately and return None. Otherwise return the encoded board
    /// of the leaf for the NN to score.
    fn select_leaf<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let mut node = 0usize;
        loop {
            let n = &self.nodes[node];
            if n.is_terminal || !n.is_expanded {
                break;
            }
            node = self.select_child(node);
        }
        if self.nodes[node].is_terminal {
            let tv = self.nodes[node].terminal_value;
            self.backpropagate(node, tv);
            self.pending_leaf = None;
            return Ok(None);
        }
        self.pending_leaf = Some(node as u32);
        let mut buf = vec![0u8; BOARD_BYTES];
        write_encoded_board(&self.states[node], buf.as_mut_slice());
        Ok(Some(PyBytes::new_bound(py, &buf)))
    }

    /// Supply NN eval for the most recently selected leaf.
    fn supply_eval(&mut self, priors: &[u8], value: f32) -> PyResult<()> {
        let Some(leaf) = self.pending_leaf else {
            return Ok(());
        };
        if priors.len() != POLICY_SIZE * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "priors must be {} bytes, got {}",
                POLICY_SIZE * 4,
                priors.len()
            )));
        }
        let priors_f32 = bytes_to_f32(priors);
        self.expand_with_policy(leaf as usize, priors_f32);
        self.backpropagate(leaf as usize, value);
        self.pending_leaf = None;
        Ok(())
    }

    /// Build the visit-based policy target, sample a move (τ-controlled),
    /// and return (policy_bytes, move_tuple, root_value). Terminal roots
    /// return zero policy + a sentinel move; callers should not act on
    /// these.
    #[pyo3(signature = (temperature=1.0, seed=0))]
    fn get_result<'py>(
        &self,
        py: Python<'py>,
        temperature: f32,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyBytes>, (i32, i32, i32, i32, Py<PyAny>), f32)> {
        let mut policy = vec![0f32; POLICY_SIZE];
        let root = &self.nodes[0];
        let mut total_visits: u64 = 0;
        for &(_, ci) in &root.children {
            total_visits += self.nodes[ci as usize].visit_count as u64;
        }
        if total_visits > 0 {
            for &(mi, ci) in &root.children {
                policy[mi as usize] =
                    self.nodes[ci as usize].visit_count as f32 / total_visits as f32;
            }
        }
        let policy_bytes = {
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(policy.as_ptr() as *const u8, policy.len() * 4)
            };
            PyBytes::new_bound(py, raw)
        };

        // Pick a move.
        let (move_info, root_value) = if root.children.is_empty() {
            (MoveInfo::default(), 0.0)
        } else {
            let mv = self.sample_move(seed, temperature);
            let root_value = if root.visit_count > 0 {
                (root.total_value / root.visit_count as f64) as f32
            } else {
                0.0
            };
            (mv, root_value)
        };

        let promo: Py<PyAny> = if move_info.promotion == 0 {
            py.None()
        } else {
            piece_type_to_str(move_info.promotion).into_py(py)
        };
        let move_tuple = (
            move_info.from_r as i32,
            move_info.from_f as i32,
            move_info.to_r as i32,
            move_info.to_f as i32,
            promo,
        );

        Ok((policy_bytes, move_tuple, root_value))
    }
}

impl MctsSearch {
    fn select_child(&self, node: usize) -> usize {
        // PUCT: q from parent-perspective (negate child's stored Q), plus
        // the c_puct * prior * sqrt(N) / (1+n) exploration term.
        //
        // FPU reduction (First-Play Urgency): unvisited children get an
        // initial Q of `parent_Q - fpu_reduction` (in parent-perspective)
        // rather than a naive 0. This prevents an over-peaked prior on a
        // single wrong move from dominating exploration — once one child
        // has been visited and its Q is known, untouched siblings start
        // from a pessimized baseline, shifting sim budget toward
        // under-prior moves when the leading move's Q disappoints.
        let n = &self.nodes[node];
        let sqrt_parent = (n.visit_count.max(1) as f32).sqrt();
        let parent_q = if n.visit_count > 0 {
            (n.total_value as f32) / n.visit_count as f32
        } else {
            0.0
        };
        let q_unvisited = parent_q - self.fpu_reduction;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = n.children[0].1 as usize;
        for &(_, ci) in &n.children {
            let c = &self.nodes[ci as usize];
            let q = if c.visit_count > 0 {
                -(c.total_value as f32) / c.visit_count as f32
            } else {
                q_unvisited
            };
            let u = self.c_puct * c.prior * sqrt_parent / (1.0 + c.visit_count as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = ci as usize;
            }
        }
        best_idx
    }

    fn backpropagate(&mut self, start: usize, value: f32) {
        let mut idx = Some(start as u32);
        let mut v = value as f64;
        while let Some(i) = idx {
            let n = &mut self.nodes[i as usize];
            n.visit_count += 1;
            n.total_value += v;
            v = -v;
            idx = if n.parent == NO_PARENT { None } else { Some(n.parent) };
        }
    }

    fn expand_with_policy(&mut self, node: usize, priors: &[f32]) {
        // Generate legal moves from this node's state.
        let (mut legal_board, mut legal_cr) = {
            let s = &self.states[node];
            (s.board, s.castling)
        };
        let (white_to_move, ep) = {
            let s = &self.states[node];
            (s.white_to_move, s.ep)
        };
        let legal = legal_moves_impl(&mut legal_board, white_to_move, &mut legal_cr, ep);

        if legal.is_empty() {
            // Dead end reached (missed by status detection at parent).
            // No legal moves = checkmate (-1 for the side to move) when
            // in check, stalemate (0) otherwise — labeling both as 0.0,
            // as an earlier version did, scores mates as draws.
            let in_check = is_in_check(&self.states[node].board, white_to_move);
            let n = &mut self.nodes[node];
            n.is_terminal = true;
            n.is_expanded = true;
            n.terminal_value = if in_check { -1.0 } else { 0.0 };
            return;
        }

        // Accumulate prior mass for the unique policy indices we'll keep
        // (collapse underpromotions onto queen-promotion slot — matches
        // Python `expand_children` + `move_to_index`).
        let is_white = white_to_move;
        let mut seen_mask = vec![false; POLICY_SIZE];
        let mut prior_sum = 0f32;
        for m in &legal {
            let mi = move_policy_index(m, is_white) as usize;
            if !seen_mask[mi] {
                seen_mask[mi] = true;
                prior_sum += priors[mi];
            }
        }
        // Reset seen for the emission pass.
        for v in seen_mask.iter_mut() {
            *v = false;
        }

        // Take a stable count for the uniform fallback when prior_sum is 0.
        let mut unique_count = 0usize;
        for m in &legal {
            let mi = move_policy_index(m, is_white) as usize;
            if !seen_mask[mi] {
                seen_mask[mi] = true;
                unique_count += 1;
            }
        }
        for v in seen_mask.iter_mut() {
            *v = false;
        }

        // Parent state snapshot for child construction.
        let (parent_board, parent_cr, parent_ep, parent_hmc, parent_fmn, parent_wtm) = {
            let s = &self.states[node];
            (
                s.board,
                s.castling,
                s.ep,
                s.hmc,
                s.fmn,
                s.white_to_move,
            )
        };

        for m in &legal {
            let mi = move_policy_index(m, is_white) as usize;
            if seen_mask[mi] {
                continue; // already emitted a child with this policy index
            }
            seen_mask[mi] = true;

            let (new_board, new_cr, new_ep, new_hmc, new_fmn, new_wtm, status) =
                apply_move_full_core(
                    parent_board,
                    parent_wtm,
                    parent_cr,
                    parent_ep,
                    parent_hmc,
                    parent_fmn,
                    m,
                );
            let child_state = NodeState {
                board: new_board,
                castling: new_cr,
                ep: new_ep,
                hmc: new_hmc,
                fmn: new_fmn,
                white_to_move: new_wtm,
                status: NodeStatus::from_str(status),
            };
            let _ = parent_board; // silence "unused" in debug if parent_board isn't used below
            let _ = parent_cr;

            let child_idx = self.nodes.len() as u32;
            let prior = if prior_sum > 0.0 {
                priors[mi] / prior_sum
            } else {
                1.0 / unique_count as f32
            };
            let (is_term, term_val) = (
                child_state.status.is_terminal(),
                child_state.status.terminal_value(),
            );
            self.states.push(child_state);
            self.nodes.push(MctsNode {
                parent: node as u32,
                children: Vec::new(),
                move_info: MoveInfo {
                    from_r: m.from_r,
                    from_f: m.from_f,
                    to_r: m.to_r,
                    to_f: m.to_f,
                    promotion: m.promotion,
                },
                visit_count: 0,
                total_value: 0.0,
                prior,
                is_expanded: is_term,
                is_terminal: is_term,
                terminal_value: term_val,
            });
            self.nodes[node].children.push((mi as u16, child_idx));
        }

        self.nodes[node].is_expanded = true;
    }

    fn add_dirichlet_noise(
        &mut self,
        node: usize,
        epsilon: f32,
        alpha: f32,
        seed: u64,
    ) {
        let children_len = self.nodes[node].children.len();
        if children_len == 0 {
            return;
        }
        let noise = dirichlet(alpha, children_len, seed);
        for (i, &(_, ci)) in self.nodes[node].children.clone().iter().enumerate() {
            let c = &mut self.nodes[ci as usize];
            c.prior = (1.0 - epsilon) * c.prior + epsilon * noise[i] as f32;
        }
    }

    fn sample_move(&self, seed: u64, temperature: f32) -> MoveInfo {
        let root = &self.nodes[0];
        // τ<=0: argmax(visit_count); ties broken by first-seen.
        if temperature <= 1e-6 {
            let mut best = &root.children[0];
            let mut best_visits = self.nodes[best.1 as usize].visit_count;
            for c in &root.children {
                let v = self.nodes[c.1 as usize].visit_count;
                if v > best_visits {
                    best = c;
                    best_visits = v;
                }
            }
            return self.nodes[best.1 as usize].move_info;
        }
        // τ>0: sample proportional to visit_count (ignore temperature
        // magnitude beyond "off"; Python's `_sample_move` does the same).
        let total_visits: u64 = root
            .children
            .iter()
            .map(|&(_, ci)| self.nodes[ci as usize].visit_count as u64)
            .sum();
        if total_visits == 0 {
            // No sims completed — uniform choice (parity with Python).
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let i = rng.gen_range(0..root.children.len());
            return self.nodes[root.children[i].1 as usize].move_info;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let r = rng.gen_range(0..total_visits);
        let mut acc: u64 = 0;
        for &(_, ci) in &root.children {
            acc += self.nodes[ci as usize].visit_count as u64;
            if r < acc {
                return self.nodes[ci as usize].move_info;
            }
        }
        // Float drift guard: fall back to highest-visits.
        let mut best = &root.children[0];
        let mut best_visits = self.nodes[best.1 as usize].visit_count;
        for c in &root.children {
            let v = self.nodes[c.1 as usize].visit_count;
            if v > best_visits {
                best = c;
                best_visits = v;
            }
        }
        self.nodes[best.1 as usize].move_info
    }
}

// ------------- helpers -------------

fn bytes_to_f32(bytes: &[u8]) -> &[f32] {
    // Caller guarantees byte length is a multiple of 4. Using a raw cast
    // keeps this a zero-copy read from the PyBytes backing buffer.
    let ptr = bytes.as_ptr() as *const f32;
    let len = bytes.len() / 4;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn move_policy_index(m: &Move, is_white: bool) -> u32 {
    let (fr, ff, tr, tf) = if is_white {
        (
            m.from_r as i32,
            m.from_f as i32,
            m.to_r as i32,
            m.to_f as i32,
        )
    } else {
        (
            7 - m.from_r as i32,
            7 - m.from_f as i32,
            7 - m.to_r as i32,
            7 - m.to_f as i32,
        )
    };
    ((fr * 8 + ff) * 64 + (tr * 8 + tf)) as u32
}

/// Write the encoded NN input (8*8*20 f32s) into `out` (must be BOARD_BYTES
/// bytes). Bit-exact with Python `encode_board`.
fn write_encoded_board(state: &NodeState, out: &mut [u8]) {
    debug_assert_eq!(out.len(), BOARD_BYTES);
    // Interpret `out` as a &mut [f32; 8*8*20] and fill.
    let fs: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, BOARD_BYTES / 4)
    };
    for v in fs.iter_mut() {
        *v = 0.0;
    }

    let white_to_move = state.white_to_move;

    // Piece planes (0..11)
    for r in 0..8usize {
        for f in 0..8usize {
            let piece = state.board[r][f];
            if piece == 0 {
                continue;
            }
            let is_white = piece > 0;
            let type_idx = (piece.unsigned_abs() as usize) - 1;
            let is_own = is_white == white_to_move;
            let channel = if is_own { type_idx } else { 6 + type_idx };
            let (or_, of_) = if white_to_move { (r, f) } else { (7 - r, 7 - f) };
            fs[plane_index(or_, of_, channel)] = 1.0;
        }
    }

    // 12: bias
    for r in 0..8 {
        for f in 0..8 {
            fs[plane_index(r, f, 12)] = 1.0;
        }
    }
    // 13: hmc / 50 clamped
    let hmc = (state.hmc as f32 / 50.0).min(1.0);
    if hmc > 0.0 {
        for r in 0..8 {
            for f in 0..8 {
                fs[plane_index(r, f, 13)] = hmc;
            }
        }
    }
    // 14: fmn / 100 clamped
    let fmn = (state.fmn as f32 / 100.0).min(1.0);
    if fmn > 0.0 {
        for r in 0..8 {
            for f in 0..8 {
                fs[plane_index(r, f, 14)] = fmn;
            }
        }
    }
    // 15..18: castling rights from stm perspective
    let rights = if white_to_move {
        [
            state.castling.wk,
            state.castling.wq,
            state.castling.bk,
            state.castling.bq,
        ]
    } else {
        [
            state.castling.bk,
            state.castling.bq,
            state.castling.wk,
            state.castling.wq,
        ]
    };
    for (i, &flag) in rights.iter().enumerate() {
        if flag {
            for r in 0..8 {
                for f in 0..8 {
                    fs[plane_index(r, f, 15 + i)] = 1.0;
                }
            }
        }
    }
    // 19: ep
    if let Some((ep_r, ep_f)) = state.ep {
        let (er, ef) = if white_to_move {
            (ep_r as usize, ep_f as usize)
        } else {
            (7 - ep_r as usize, 7 - ep_f as usize)
        };
        fs[plane_index(er, ef, 19)] = 1.0;
    }
}

fn dirichlet(alpha: f32, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let gamma = Gamma::new(alpha as f64, 1.0).expect("alpha>0");
    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / n as f64; n];
    }
    samples.into_iter().map(|x| x / sum).collect()
}

fn parse_state_dict(state_dict: &Bound<'_, PyDict>) -> PyResult<NodeState> {
    use pyo3::types::PyList;
    let board_obj = state_dict
        .get_item("board")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("board"))?;
    let board_list = board_obj.downcast::<PyList>()?;
    let board = pack_board(board_list)?;

    let turn_obj = state_dict
        .get_item("currentTurn")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("currentTurn"))?;
    let turn_str: String = turn_obj.extract()?;
    let white_to_move = turn_str == "white";

    let cr_obj = state_dict
        .get_item("castlingRights")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("castlingRights"))?;
    let castling = pack_castling(cr_obj.downcast::<PyDict>()?)?;

    let ep_obj = state_dict
        .get_item("enPassantTarget")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("enPassantTarget"))?;
    let ep = pack_en_passant(&ep_obj)?;

    let hmc: i32 = state_dict
        .get_item("halfMoveClock")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("halfMoveClock"))?
        .extract()?;
    let fmn: i32 = state_dict
        .get_item("fullMoveNumber")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("fullMoveNumber"))?
        .extract()?;

    // `status` may be "active" / "check" / "checkmate" / "stalemate" /
    // "draw" / "waiting". For "waiting" (fresh root with status not yet
    // computed), we recompute here — Python's MCTSSearch does the same.
    let status_obj = state_dict.get_item("status")?;
    let status_str: String = match status_obj {
        Some(obj) => obj.extract().unwrap_or_else(|_| "active".to_string()),
        None => "active".to_string(),
    };

    let status = if status_str == "waiting" || status_str.is_empty() {
        // Recompute from scratch using engine helpers.
        let mut b = board;
        let mut cr = castling;
        let legal = legal_moves_impl(&mut b, white_to_move, &mut cr, ep);
        let in_check = is_in_check(&board, white_to_move);
        if legal.is_empty() {
            if in_check {
                NodeStatus::Checkmate
            } else {
                NodeStatus::Stalemate
            }
        } else if in_check {
            NodeStatus::Check
        } else if hmc >= 100 {
            NodeStatus::Draw
        } else {
            NodeStatus::Active
        }
    } else {
        NodeStatus::from_str(&status_str)
    };

    // Silence unused-import warning in release builds when PAWN isn't
    // referenced from this file's body directly.
    let _ = PAWN;

    Ok(NodeState {
        board,
        castling,
        ep,
        hmc,
        fmn,
        white_to_move,
        status,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MctsSearch>()?;
    Ok(())
}
