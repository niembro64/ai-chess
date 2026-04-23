//! Rust port of the Python chess engine.
//!
//! Exported to Python via PyO3 as the `chess_ai_rust` module. Python callers
//! pass boards as nested lists of `{"color", "type"}` dicts (matching
//! `ChessGameState.to_dict()["board"]`) and get back tuples describing
//! moves; the engine itself operates on a packed `i8`-per-square board.
//!
//! Piece encoding:
//!     0 = empty
//!     sign = colour (positive = white, negative = black)
//!     magnitude = piece type, in the same order as Python's `_PIECE_TYPES`:
//!         1 king, 2 queen, 3 rook, 4 bishop, 5 knight, 6 pawn.
//! This aligns with the Python side: `piece_type_index = abs(piece) - 1`.
//!
//! Parity with the Python engine (and transitively with the TS engine) is
//! enforced by `tests/test_rust_engine_parity.py`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

mod mcts;

// ------------------------------------------------------------
// Constants / types
// ------------------------------------------------------------

pub(crate) const KING: i8 = 1;
pub(crate) const QUEEN: i8 = 2;
pub(crate) const ROOK: i8 = 3;
pub(crate) const BISHOP: i8 = 4;
pub(crate) const KNIGHT: i8 = 5;
pub(crate) const PAWN: i8 = 6;

// Must match Python _KNIGHT_OFFSETS / _ROOK_DIRS / _BISHOP_DIRS order to
// preserve legal-move ordering for parity tests.
const KNIGHT_OFFSETS: [(i32, i32); 8] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
];
const ROOK_DIRS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const QUEEN_DIRS: [(i32, i32); 8] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
];

pub(crate) type Board = [[i8; 8]; 8];

#[derive(Clone, Copy, Debug)]
pub(crate) struct CastlingRights {
    pub(crate) wk: bool,
    pub(crate) wq: bool,
    pub(crate) bk: bool,
    pub(crate) bq: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Move {
    pub(crate) from_r: u8,
    pub(crate) from_f: u8,
    pub(crate) to_r: u8,
    pub(crate) to_f: u8,
    // 0 = no promotion; otherwise one of QUEEN/ROOK/BISHOP/KNIGHT.
    pub(crate) promotion: i8,
}

// ------------------------------------------------------------
// Attack detection
// ------------------------------------------------------------

#[inline]
fn is_pos_attacked_by(board: &Board, r0: i32, f0: i32, by_white: bool) -> bool {
    // Knight
    for (dr, df) in KNIGHT_OFFSETS.iter() {
        let tr = r0 + dr;
        let tf = f0 + df;
        if tr >= 0 && tr < 8 && tf >= 0 && tf < 8 {
            let p = board[tr as usize][tf as usize];
            if p.abs() == KNIGHT && (p > 0) == by_white {
                return true;
            }
        }
    }

    // Pawn: a square at r0 is pawn-attacked from r0 + pawn_dir ± 1 file.
    let pawn_dir: i32 = if by_white { 1 } else { -1 };
    let tr = r0 + pawn_dir;
    if tr >= 0 && tr < 8 {
        for df in [-1i32, 1i32].iter() {
            let tf = f0 + df;
            if tf >= 0 && tf < 8 {
                let p = board[tr as usize][tf as usize];
                if p.abs() == PAWN && (p > 0) == by_white {
                    return true;
                }
            }
        }
    }

    // King (adjacent squares)
    for dr in -1i32..=1 {
        let tr = r0 + dr;
        if tr < 0 || tr >= 8 { continue; }
        for df in -1i32..=1 {
            if dr == 0 && df == 0 { continue; }
            let tf = f0 + df;
            if tf < 0 || tf >= 8 { continue; }
            let p = board[tr as usize][tf as usize];
            if p.abs() == KING && (p > 0) == by_white {
                return true;
            }
        }
    }

    // Rook / Queen rays
    for (dr, df) in ROOK_DIRS.iter() {
        let mut tr = r0 + dr;
        let mut tf = f0 + df;
        while tr >= 0 && tr < 8 && tf >= 0 && tf < 8 {
            let p = board[tr as usize][tf as usize];
            if p != 0 {
                if (p > 0) == by_white && (p.abs() == ROOK || p.abs() == QUEEN) {
                    return true;
                }
                break;
            }
            tr += dr;
            tf += df;
        }
    }

    // Bishop / Queen rays
    for (dr, df) in BISHOP_DIRS.iter() {
        let mut tr = r0 + dr;
        let mut tf = f0 + df;
        while tr >= 0 && tr < 8 && tf >= 0 && tf < 8 {
            let p = board[tr as usize][tf as usize];
            if p != 0 {
                if (p > 0) == by_white && (p.abs() == BISHOP || p.abs() == QUEEN) {
                    return true;
                }
                break;
            }
            tr += dr;
            tf += df;
        }
    }

    false
}

fn find_king(board: &Board, white: bool) -> (i32, i32) {
    for r in 0..8 {
        for f in 0..8 {
            let p = board[r][f];
            if p.abs() == KING && (p > 0) == white {
                return (r as i32, f as i32);
            }
        }
    }
    panic!("king not found for {}", if white { "white" } else { "black" });
}

#[inline]
pub(crate) fn is_in_check(board: &Board, white: bool) -> bool {
    let (kr, kf) = find_king(board, white);
    is_pos_attacked_by(board, kr, kf, !white)
}

// ------------------------------------------------------------
// Pseudo-legal move generation (matches Python order byte-for-byte)
// ------------------------------------------------------------

fn pseudo_legal_moves(
    board: &Board,
    white_to_move: bool,
    castling: &CastlingRights,
    en_passant: Option<(u8, u8)>,
) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);

    for rank in 0..8usize {
        for file in 0..8usize {
            let piece = board[rank][file];
            if piece == 0 { continue; }
            if (piece > 0) != white_to_move { continue; }
            let pt = piece.abs();
            let from_r = rank as u8;
            let from_f = file as u8;

            match pt {
                PAWN => {
                    let direction: i32 = if white_to_move { -1 } else { 1 };
                    let start_rank: i32 = if white_to_move { 6 } else { 1 };
                    let promo_rank: i32 = if white_to_move { 0 } else { 7 };

                    // Forward one
                    let fr = rank as i32 + direction;
                    if fr >= 0 && fr < 8 && board[fr as usize][file] == 0 {
                        if fr == promo_rank {
                            for &promo in &[QUEEN, ROOK, BISHOP, KNIGHT] {
                                moves.push(Move { from_r, from_f, to_r: fr as u8, to_f: file as u8, promotion: promo });
                            }
                        } else {
                            moves.push(Move { from_r, from_f, to_r: fr as u8, to_f: file as u8, promotion: 0 });
                        }

                        if rank as i32 == start_rank {
                            let fr2 = rank as i32 + 2 * direction;
                            if board[fr2 as usize][file] == 0 {
                                moves.push(Move { from_r, from_f, to_r: fr2 as u8, to_f: file as u8, promotion: 0 });
                            }
                        }
                    }

                    // Captures (incl. en passant)
                    for &df in &[-1i32, 1i32] {
                        let cr = rank as i32 + direction;
                        let cf = file as i32 + df;
                        if cr < 0 || cr >= 8 || cf < 0 || cf >= 8 { continue; }
                        let target = board[cr as usize][cf as usize];
                        if target != 0 && (target > 0) != white_to_move {
                            if cr == promo_rank {
                                for &promo in &[QUEEN, ROOK, BISHOP, KNIGHT] {
                                    moves.push(Move { from_r, from_f, to_r: cr as u8, to_f: cf as u8, promotion: promo });
                                }
                            } else {
                                moves.push(Move { from_r, from_f, to_r: cr as u8, to_f: cf as u8, promotion: 0 });
                            }
                        }
                        if let Some((er, ef)) = en_passant {
                            if cr as u8 == er && cf as u8 == ef {
                                moves.push(Move { from_r, from_f, to_r: cr as u8, to_f: cf as u8, promotion: 0 });
                            }
                        }
                    }
                }

                KNIGHT => {
                    for &(dr, df) in KNIGHT_OFFSETS.iter() {
                        let tr = rank as i32 + dr;
                        let tf = file as i32 + df;
                        if tr < 0 || tr >= 8 || tf < 0 || tf >= 8 { continue; }
                        let target = board[tr as usize][tf as usize];
                        if target == 0 || (target > 0) != white_to_move {
                            moves.push(Move { from_r, from_f, to_r: tr as u8, to_f: tf as u8, promotion: 0 });
                        }
                    }
                }

                BISHOP | ROOK | QUEEN => {
                    let dirs: &[(i32, i32)] = match pt {
                        BISHOP => &BISHOP_DIRS,
                        ROOK => &ROOK_DIRS,
                        _ => &QUEEN_DIRS,
                    };
                    for &(dr, df) in dirs.iter() {
                        for i in 1..8 {
                            let tr = rank as i32 + dr * i;
                            let tf = file as i32 + df * i;
                            if tr < 0 || tr >= 8 || tf < 0 || tf >= 8 { break; }
                            let target = board[tr as usize][tf as usize];
                            if target == 0 {
                                moves.push(Move { from_r, from_f, to_r: tr as u8, to_f: tf as u8, promotion: 0 });
                            } else {
                                if (target > 0) != white_to_move {
                                    moves.push(Move { from_r, from_f, to_r: tr as u8, to_f: tf as u8, promotion: 0 });
                                }
                                break;
                            }
                        }
                    }
                }

                KING => {
                    // Adjacent squares. Iterate (dr, df) in (-1..=1, -1..=1) order,
                    // skipping (0, 0) — matches Python.
                    for dr in -1i32..=1 {
                        for df in -1i32..=1 {
                            if dr == 0 && df == 0 { continue; }
                            let tr = rank as i32 + dr;
                            let tf = file as i32 + df;
                            if tr < 0 || tr >= 8 || tf < 0 || tf >= 8 { continue; }
                            let target = board[tr as usize][tf as usize];
                            if target == 0 || (target > 0) != white_to_move {
                                moves.push(Move { from_r, from_f, to_r: tr as u8, to_f: tf as u8, promotion: 0 });
                            }
                        }
                    }

                    // Castling — exact same gating as Python.
                    if white_to_move && rank == 7 && file == 4 {
                        if castling.wk
                            && board[7][5] == 0 && board[7][6] == 0
                            && board[7][7] == ROOK
                            && !is_pos_attacked_by(board, 7, 4, false)
                            && !is_pos_attacked_by(board, 7, 5, false)
                            && !is_pos_attacked_by(board, 7, 6, false)
                        {
                            moves.push(Move { from_r: 7, from_f: 4, to_r: 7, to_f: 6, promotion: 0 });
                        }
                        if castling.wq
                            && board[7][3] == 0 && board[7][2] == 0 && board[7][1] == 0
                            && board[7][0] == ROOK
                            && !is_pos_attacked_by(board, 7, 4, false)
                            && !is_pos_attacked_by(board, 7, 3, false)
                            && !is_pos_attacked_by(board, 7, 2, false)
                        {
                            moves.push(Move { from_r: 7, from_f: 4, to_r: 7, to_f: 2, promotion: 0 });
                        }
                    }
                    if !white_to_move && rank == 0 && file == 4 {
                        if castling.bk
                            && board[0][5] == 0 && board[0][6] == 0
                            && board[0][7] == -ROOK
                            && !is_pos_attacked_by(board, 0, 4, true)
                            && !is_pos_attacked_by(board, 0, 5, true)
                            && !is_pos_attacked_by(board, 0, 6, true)
                        {
                            moves.push(Move { from_r: 0, from_f: 4, to_r: 0, to_f: 6, promotion: 0 });
                        }
                        if castling.bq
                            && board[0][3] == 0 && board[0][2] == 0 && board[0][1] == 0
                            && board[0][0] == -ROOK
                            && !is_pos_attacked_by(board, 0, 4, true)
                            && !is_pos_attacked_by(board, 0, 3, true)
                            && !is_pos_attacked_by(board, 0, 2, true)
                        {
                            moves.push(Move { from_r: 0, from_f: 4, to_r: 0, to_f: 2, promotion: 0 });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    moves
}

// ------------------------------------------------------------
// In-place apply (with the info needed to undo)
// ------------------------------------------------------------

pub(crate) fn apply_move_to_board(board: &mut Board, m: &Move, castling: &mut CastlingRights) -> (i8, bool, bool) {
    let piece = board[m.from_r as usize][m.from_f as usize];
    let captured = board[m.to_r as usize][m.to_f as usize];
    let mut is_en_passant = false;
    let mut is_castle = false;

    // En passant capture
    if piece.abs() == PAWN && m.to_f != m.from_f && captured == 0 {
        is_en_passant = true;
        board[m.from_r as usize][m.to_f as usize] = 0;
    }

    // Castling
    if piece.abs() == KING && (m.to_f as i32 - m.from_f as i32).abs() == 2 {
        is_castle = true;
        if m.to_f == 6 {
            board[m.from_r as usize][5] = board[m.from_r as usize][7];
            board[m.from_r as usize][7] = 0;
        } else {
            board[m.from_r as usize][3] = board[m.from_r as usize][0];
            board[m.from_r as usize][0] = 0;
        }
    }

    // Move piece
    board[m.to_r as usize][m.to_f as usize] = piece;
    board[m.from_r as usize][m.from_f as usize] = 0;

    // Promotion
    if m.promotion != 0 {
        let sign: i8 = if piece > 0 { 1 } else { -1 };
        board[m.to_r as usize][m.to_f as usize] = sign * m.promotion;
    }

    // Castling rights updates
    if piece.abs() == KING {
        if piece > 0 { castling.wk = false; castling.wq = false; }
        else { castling.bk = false; castling.bq = false; }
    }
    if piece.abs() == ROOK {
        if piece > 0 {
            if m.from_r == 7 && m.from_f == 7 { castling.wk = false; }
            if m.from_r == 7 && m.from_f == 0 { castling.wq = false; }
        } else {
            if m.from_r == 0 && m.from_f == 7 { castling.bk = false; }
            if m.from_r == 0 && m.from_f == 0 { castling.bq = false; }
        }
    }
    // Rook captured on its starting square
    if m.to_r == 0 && m.to_f == 7 { castling.bk = false; }
    if m.to_r == 0 && m.to_f == 0 { castling.bq = false; }
    if m.to_r == 7 && m.to_f == 7 { castling.wk = false; }
    if m.to_r == 7 && m.to_f == 0 { castling.wq = false; }

    (captured, is_en_passant, is_castle)
}

// ------------------------------------------------------------
// get_legal_moves — the headline hot path
// ------------------------------------------------------------

pub(crate) fn legal_moves_impl(
    board: &mut Board,
    white_to_move: bool,
    castling: &mut CastlingRights,
    en_passant: Option<(u8, u8)>,
) -> Vec<Move> {
    let pseudo = pseudo_legal_moves(board, white_to_move, castling, en_passant);

    let saved_castling = *castling;
    let (king_r0, king_f0) = find_king(board, white_to_move);

    let mut legal = Vec::with_capacity(pseudo.len());

    for m in &pseudo {
        let sq_from = board[m.from_r as usize][m.from_f as usize];
        let sq_to = board[m.to_r as usize][m.to_f as usize];

        let is_pawn = sq_from != 0 && sq_from.abs() == PAWN;
        let is_king_move = sq_from != 0 && sq_from.abs() == KING;
        let is_en_passant = is_pawn && m.from_f != m.to_f && sq_to == 0;
        let is_castle = is_king_move && (m.to_f as i32 - m.from_f as i32).abs() == 2;

        let sq_ep = if is_en_passant {
            board[m.from_r as usize][m.to_f as usize]
        } else { 0 };

        let (rook_from_f, rook_to_f, sq_rook_from, sq_rook_to) = if is_castle {
            let rf = if m.to_f == 6 { 7usize } else { 0usize };
            let rt = if m.to_f == 6 { 5usize } else { 3usize };
            (rf, rt, board[m.from_r as usize][rf], board[m.from_r as usize][rt])
        } else { (0, 0, 0, 0) };

        // Apply in place
        apply_move_to_board(board, m, castling);

        // King position after the move
        let (k_r, k_f) = if is_king_move {
            (m.to_r as i32, m.to_f as i32)
        } else {
            (king_r0, king_f0)
        };

        if !is_pos_attacked_by(board, k_r, k_f, !white_to_move) {
            legal.push(*m);
        }

        // Undo
        board[m.from_r as usize][m.from_f as usize] = sq_from;
        board[m.to_r as usize][m.to_f as usize] = sq_to;
        if is_en_passant {
            board[m.from_r as usize][m.to_f as usize] = sq_ep;
        }
        if is_castle {
            board[m.from_r as usize][rook_from_f] = sq_rook_from;
            board[m.from_r as usize][rook_to_f] = sq_rook_to;
        }
        *castling = saved_castling;
    }

    legal
}

// ------------------------------------------------------------
// Python ↔ Rust marshalling
// ------------------------------------------------------------

pub(crate) fn piece_type_from_str(s: &str) -> PyResult<i8> {
    Ok(match s {
        "king" => KING,
        "queen" => QUEEN,
        "rook" => ROOK,
        "bishop" => BISHOP,
        "knight" => KNIGHT,
        "pawn" => PAWN,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown piece type: {}", s))),
    })
}

pub(crate) fn piece_type_to_str(pt: i8) -> &'static str {
    match pt {
        KING => "king",
        QUEEN => "queen",
        ROOK => "rook",
        BISHOP => "bishop",
        KNIGHT => "knight",
        PAWN => "pawn",
        _ => "pawn", // unreachable
    }
}

pub(crate) fn pack_board(py_board: &Bound<'_, PyList>) -> PyResult<Board> {
    let mut board: Board = [[0i8; 8]; 8];
    for (r, row_any) in py_board.iter().enumerate() {
        if r >= 8 { break; }
        let row = row_any.downcast::<PyList>()?;
        for (f, cell_any) in row.iter().enumerate() {
            if f >= 8 { break; }
            if cell_any.is_none() {
                board[r][f] = 0;
            } else {
                let cell = cell_any.downcast::<PyDict>()?;
                let color_obj = cell.get_item("color")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("color"))?;
                let type_obj = cell.get_item("type")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("type"))?;
                let color: String = color_obj.extract()?;
                let type_str: String = type_obj.extract()?;
                let pt = piece_type_from_str(&type_str)?;
                let sign: i8 = if color == "white" { 1 } else { -1 };
                board[r][f] = sign * pt;
            }
        }
    }
    Ok(board)
}

pub(crate) fn pack_castling(py_castling: &Bound<'_, PyDict>) -> PyResult<CastlingRights> {
    let get = |k: &str| -> PyResult<bool> {
        let obj = py_castling.get_item(k)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(k.to_string()))?;
        obj.extract()
    };
    Ok(CastlingRights {
        wk: get("whiteKingside")?,
        wq: get("whiteQueenside")?,
        bk: get("blackKingside")?,
        bq: get("blackQueenside")?,
    })
}

pub(crate) fn pack_en_passant(py_ep: &Bound<'_, PyAny>) -> PyResult<Option<(u8, u8)>> {
    if py_ep.is_none() {
        return Ok(None);
    }
    let d = py_ep.downcast::<PyDict>()?;
    let rank: i32 = d.get_item("rank")?.unwrap().extract()?;
    let file: i32 = d.get_item("file")?.unwrap().extract()?;
    Ok(Some((rank as u8, file as u8)))
}

// ------------------------------------------------------------
// PyO3 entry points
// ------------------------------------------------------------

/// Return a list of (from_rank, from_file, to_rank, to_file, promotion_str_or_None)
/// tuples. Ordering matches the Python engine byte-for-byte.
#[pyfunction]
fn get_legal_moves(
    py: Python<'_>,
    board_list: Bound<'_, PyList>,
    current_turn: &str,
    castling: Bound<'_, PyDict>,
    en_passant: Bound<'_, PyAny>,
) -> PyResult<Py<PyList>> {
    let mut board = pack_board(&board_list)?;
    let mut cr = pack_castling(&castling)?;
    let ep = pack_en_passant(&en_passant)?;
    let white_to_move = current_turn == "white";

    let moves = legal_moves_impl(&mut board, white_to_move, &mut cr, ep);

    let out = PyList::empty_bound(py);
    for m in &moves {
        let promo: Py<PyAny> = if m.promotion == 0 {
            py.None()
        } else {
            piece_type_to_str(m.promotion).into_py(py)
        };
        let tup = (m.from_r as i32, m.from_f as i32, m.to_r as i32, m.to_f as i32, promo);
        out.append(tup)?;
    }
    Ok(out.unbind())
}

#[pyfunction]
fn is_in_check_py(board_list: Bound<'_, PyList>, color: &str) -> PyResult<bool> {
    let board = pack_board(&board_list)?;
    Ok(is_in_check(&board, color == "white"))
}

/// Full `apply_move` port: accepts the same state fields `ChessGameState`
/// holds, applies the move in-place on a local board, runs the status
/// detection (checkmate / stalemate / check / draw / active) via the
/// Rust engine, and returns all the fields the Python wrapper needs to
/// rebuild a new `ChessGameState`. The board is returned as a flat list
/// of 64 ints (signed piece codes: +/- {1..6}, 0 = empty) which the
/// Python wrapper maps back to `Piece` dataclass instances via a static
/// lookup table — significantly cheaper than round-tripping through
/// dict-of-dicts.
/// Pure-Rust state transition: apply a move and detect the resulting
/// status. Shared between `apply_move_full` (Python-facing) and the
/// MCTS tree expansion (which builds hundreds of child states per sim
/// without crossing FFI). Returns
///   (new_board, new_castling, new_ep, new_hmc, new_fmn, new_white_to_move, status)
/// where `status` matches Python's status strings exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_move_full_core(
    mut board: Board,
    white_to_move: bool,
    mut castling: CastlingRights,
    _ep: Option<(u8, u8)>,
    hmc: i32,
    fmn: i32,
    m: &Move,
) -> (
    Board,
    CastlingRights,
    Option<(u8, u8)>,
    i32,
    i32,
    bool,
    &'static str,
) {
    let piece_before = board[m.from_r as usize][m.from_f as usize];
    let is_pawn_move = piece_before != 0 && piece_before.abs() == PAWN;
    let was_capture = board[m.to_r as usize][m.to_f as usize] != 0;

    let (_, is_en_passant, _) = apply_move_to_board(&mut board, m, &mut castling);

    let new_ep: Option<(u8, u8)> =
        if is_pawn_move && (m.to_r as i32 - m.from_r as i32).abs() == 2 {
            Some(((m.from_r + m.to_r) / 2, m.from_f))
        } else {
            None
        };

    let new_hmc = if is_pawn_move || was_capture || is_en_passant {
        0
    } else {
        hmc + 1
    };

    let new_fmn = if !white_to_move { fmn + 1 } else { fmn };
    let new_white_to_move = !white_to_move;

    // Status detection (matches apply_move_full): count legal moves on a
    // scratch copy and combine with in-check detection.
    let mut test_board = board;
    let mut test_cr = castling;
    let next_legal = legal_moves_impl(&mut test_board, new_white_to_move, &mut test_cr, new_ep);
    let in_check = is_in_check(&board, new_white_to_move);
    let status: &'static str = if next_legal.is_empty() {
        if in_check { "checkmate" } else { "stalemate" }
    } else if in_check {
        "check"
    } else if new_hmc >= 100 {
        "draw"
    } else {
        "active"
    };

    (
        board,
        castling,
        new_ep,
        new_hmc,
        new_fmn,
        new_white_to_move,
        status,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    board_list, current_turn, castling, en_passant,
    half_move_clock, full_move_number,
    from_r, from_f, to_r, to_f, promotion=None,
))]
fn apply_move_full(
    py: Python<'_>,
    board_list: Bound<'_, PyList>,
    current_turn: &str,
    castling: Bound<'_, PyDict>,
    en_passant: Bound<'_, PyAny>,
    half_move_clock: i32,
    full_move_number: i32,
    from_r: i32,
    from_f: i32,
    to_r: i32,
    to_f: i32,
    promotion: Option<String>,
) -> PyResult<Py<PyDict>> {
    let board = pack_board(&board_list)?;
    let cr = pack_castling(&castling)?;
    let ep = pack_en_passant(&en_passant)?;
    let white_to_move = current_turn == "white";

    let promo: i8 = match promotion.as_deref() {
        Some(s) => piece_type_from_str(s)?,
        None => 0,
    };
    let m = Move {
        from_r: from_r as u8,
        from_f: from_f as u8,
        to_r: to_r as u8,
        to_f: to_f as u8,
        promotion: promo,
    };

    if board[m.from_r as usize][m.from_f as usize] == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No piece at source square",
        ));
    }

    let (board, cr, new_ep, new_hmc, new_fmn, new_white_to_move, status) =
        apply_move_full_core(board, white_to_move, cr, ep, half_move_clock, full_move_number, &m);

    // Pack new state. Board goes out as a flat list of 64 signed ints
    // — the Python wrapper maps each int back to its cached Piece via a
    // dict lookup.
    let result = PyDict::new_bound(py);

    let board_flat = PyList::empty_bound(py);
    for r in 0..8 {
        for f in 0..8 {
            board_flat.append(board[r][f] as i32)?;
        }
    }
    result.set_item("board", board_flat)?;

    result.set_item(
        "currentTurn",
        if new_white_to_move { "white" } else { "black" },
    )?;

    let cr_dict = PyDict::new_bound(py);
    cr_dict.set_item("whiteKingside", cr.wk)?;
    cr_dict.set_item("whiteQueenside", cr.wq)?;
    cr_dict.set_item("blackKingside", cr.bk)?;
    cr_dict.set_item("blackQueenside", cr.bq)?;
    result.set_item("castlingRights", cr_dict)?;

    match new_ep {
        Some((r, f)) => {
            let ep_dict = PyDict::new_bound(py);
            ep_dict.set_item("rank", r as i32)?;
            ep_dict.set_item("file", f as i32)?;
            result.set_item("enPassantTarget", ep_dict)?;
        }
        None => {
            result.set_item("enPassantTarget", py.None())?;
        }
    }

    result.set_item("halfMoveClock", new_hmc)?;
    result.set_item("fullMoveNumber", new_fmn)?;
    result.set_item("status", status)?;

    Ok(result.unbind())
}

#[pyfunction]
fn is_square_attacked_by_py(
    board_list: Bound<'_, PyList>,
    rank: i32,
    file: i32,
    by_color: &str,
) -> PyResult<bool> {
    let board = pack_board(&board_list)?;
    Ok(is_pos_attacked_by(&board, rank, file, by_color == "white"))
}

// ------------------------------------------------------------
// Module
// ------------------------------------------------------------

/// Generate every `(move, child_state)` pair from the given position in a
/// single FFI call.
///
/// MCTS tree expansion used to make one `get_legal_moves` call + N
/// (~30 avg) separate `apply_move` calls per leaf, each paying ~20 µs
/// of board-packing and state-dict construction to cross the Python/Rust
/// boundary. This function collapses all of that into one round-trip: we
/// pack the parent board once, enumerate legal moves, apply each in a
/// cloned scratch board, compute the full child state (including status),
/// and return a list of `(move_tuple, child_state_dict)` pairs.
///
/// Parity with composition of get_legal_moves + apply_move_full is
/// transitive (both already parity-tested against the pure-Python engine
/// and the TS fixture) but a dedicated test covers the combined path.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generate_children(
    py: Python<'_>,
    board_list: Bound<'_, PyList>,
    current_turn: &str,
    castling: Bound<'_, PyDict>,
    en_passant: Bound<'_, PyAny>,
    half_move_clock: i32,
    full_move_number: i32,
) -> PyResult<Py<PyList>> {
    let mut board = pack_board(&board_list)?;
    let mut cr = pack_castling(&castling)?;
    let ep = pack_en_passant(&en_passant)?;
    let white_to_move = current_turn == "white";

    let legal_moves = legal_moves_impl(&mut board, white_to_move, &mut cr, ep);

    let new_white_to_move = !white_to_move;
    let new_fmn = if !white_to_move {
        full_move_number + 1
    } else {
        full_move_number
    };

    let out = PyList::empty_bound(py);

    for m in &legal_moves {
        // Each child starts from a pristine copy of the parent board + castling.
        let mut child_board = board;
        let mut child_cr = cr;

        let piece_before = child_board[m.from_r as usize][m.from_f as usize];
        let is_pawn = piece_before.abs() == PAWN;
        let was_capture = child_board[m.to_r as usize][m.to_f as usize] != 0;

        let (_, is_en_passant, _) = apply_move_to_board(&mut child_board, m, &mut child_cr);

        let new_ep: Option<(u8, u8)> =
            if is_pawn && (m.to_r as i32 - m.from_r as i32).abs() == 2 {
                Some(((m.from_r + m.to_r) / 2, m.from_f))
            } else {
                None
            };

        let new_hmc = if is_pawn || was_capture || is_en_passant {
            0
        } else {
            half_move_clock + 1
        };

        // Status: run a legal-moves check on the child.
        let mut test_board = child_board;
        let mut test_cr = child_cr;
        let next_legal =
            legal_moves_impl(&mut test_board, new_white_to_move, &mut test_cr, new_ep);
        let in_check = is_in_check(&child_board, new_white_to_move);
        let status: &str = if next_legal.is_empty() {
            if in_check { "checkmate" } else { "stalemate" }
        } else if in_check {
            "check"
        } else if new_hmc >= 100 {
            "draw"
        } else {
            "active"
        };

        // Flat child dict: move fields (move_*) + state fields (board, currentTurn, ...).
        // Flat rather than nested so we don't fight the PyO3 0.22 boxed-object
        // conversion helpers when zipping two objects into a single result item.
        let child = PyDict::new_bound(py);

        child.set_item("move_from_r", m.from_r as i32)?;
        child.set_item("move_from_f", m.from_f as i32)?;
        child.set_item("move_to_r", m.to_r as i32)?;
        child.set_item("move_to_f", m.to_f as i32)?;
        if m.promotion == 0 {
            child.set_item("move_promotion", py.None())?;
        } else {
            child.set_item("move_promotion", piece_type_to_str(m.promotion))?;
        }

        let board_flat = PyList::empty_bound(py);
        for r in 0..8 {
            for f in 0..8 {
                board_flat.append(child_board[r][f] as i32)?;
            }
        }
        child.set_item("board", board_flat)?;
        child.set_item(
            "currentTurn",
            if new_white_to_move { "white" } else { "black" },
        )?;

        let cr_dict = PyDict::new_bound(py);
        cr_dict.set_item("whiteKingside", child_cr.wk)?;
        cr_dict.set_item("whiteQueenside", child_cr.wq)?;
        cr_dict.set_item("blackKingside", child_cr.bk)?;
        cr_dict.set_item("blackQueenside", child_cr.bq)?;
        child.set_item("castlingRights", cr_dict)?;

        match new_ep {
            Some((r, f)) => {
                let ep_d = PyDict::new_bound(py);
                ep_d.set_item("rank", r as i32)?;
                ep_d.set_item("file", f as i32)?;
                child.set_item("enPassantTarget", ep_d)?;
            }
            None => {
                child.set_item("enPassantTarget", py.None())?;
            }
        }
        child.set_item("halfMoveClock", new_hmc)?;
        child.set_item("fullMoveNumber", new_fmn)?;
        child.set_item("status", status)?;

        out.append(child)?;
    }

    Ok(out.unbind())
}

// ------------------------------------------------------------
// Board encoding (NN input)
// ------------------------------------------------------------
//
// Mirrors `chess_ai.encoding.encode_board` bit-for-bit. Returns bytes
// holding a flat [8*8*NUM_PLANES] f32 row-major tensor; Python wraps
// them with np.frombuffer. Avoids the per-square Python loop that
// dominated encoding time on every MCTS leaf.
//
// Channel layout (NUM_PLANES = 20):
//   0..5   own pieces    (king, queen, rook, bishop, knight, pawn)
//   6..11  opponent pieces (same order)
//   12     constant 1.0 (bias)
//   13     halfMoveClock / 50 (clamped)
//   14     fullMoveNumber / 100 (clamped)
//   15..18 castling rights (from stm perspective)
//   19     en passant target square
//
// For black-to-move positions, rank/file are flipped so "own side"
// always occupies the bottom half of the board — matching Python.

pub(crate) const NUM_PLANES: usize = 20;

#[inline(always)]
pub(crate) fn plane_index(r: usize, f: usize, c: usize) -> usize {
    (r * 8 + f) * NUM_PLANES + c
}

pub(crate) fn fill_constant_plane(data: &mut [f32], channel: usize, value: f32) {
    for r in 0..8 {
        for f in 0..8 {
            data[plane_index(r, f, channel)] = value;
        }
    }
}

/// Encode a board to the flat NN-input tensor. Returned as `bytes`
/// holding `8*8*20 = 1280` little-endian f32s; Python wraps via
/// `np.frombuffer(..., dtype=np.float32)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn encode_board(
    py: Python<'_>,
    board_list: Bound<'_, PyList>,
    current_turn: &str,
    half_move_clock: i32,
    full_move_number: i32,
    castling: Bound<'_, PyDict>,
    en_passant: Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    let board = pack_board(&board_list)?;
    let cr = pack_castling(&castling)?;
    let ep = pack_en_passant(&en_passant)?;
    let white_to_move = current_turn == "white";

    let mut data = vec![0f32; 8 * 8 * NUM_PLANES];

    // Piece planes (0..11). Mirror coords when stm is black so "own side"
    // occupies the bottom half of the planes.
    for r in 0..8 {
        for f in 0..8 {
            let piece = board[r][f];
            if piece == 0 {
                continue;
            }
            let is_white = piece > 0;
            let type_idx = (piece.unsigned_abs() as usize) - 1; // king=0 .. pawn=5
            let is_own = is_white == white_to_move;
            let channel = if is_own { type_idx } else { 6 + type_idx };
            let (or, of) = if white_to_move { (r, f) } else { (7 - r, 7 - f) };
            data[plane_index(or, of, channel)] = 1.0;
        }
    }

    // Channel 12: constant bias.
    fill_constant_plane(&mut data, 12, 1.0);

    // Channel 13: halfMoveClock / 50, clamped to 1.0. Skip the write if
    // the value is zero to match the Python behavior (which conditionally
    // assigns).
    let hmc = (half_move_clock as f32 / 50.0).min(1.0);
    if hmc > 0.0 {
        fill_constant_plane(&mut data, 13, hmc);
    }

    // Channel 14: fullMoveNumber / 100, clamped.
    let fmn = (full_move_number as f32 / 100.0).min(1.0);
    if fmn > 0.0 {
        fill_constant_plane(&mut data, 14, fmn);
    }

    // Channels 15..18: castling rights from stm perspective. White-to-move
    // gets (wk, wq, bk, bq); black-to-move gets (bk, bq, wk, wq).
    let rights = if white_to_move {
        [cr.wk, cr.wq, cr.bk, cr.bq]
    } else {
        [cr.bk, cr.bq, cr.wk, cr.wq]
    };
    for (i, &flag) in rights.iter().enumerate() {
        if flag {
            fill_constant_plane(&mut data, 15 + i, 1.0);
        }
    }

    // Channel 19: en passant target (single one-hot square).
    if let Some((ep_r, ep_f)) = ep {
        let (er, ef) = if white_to_move {
            (ep_r as usize, ep_f as usize)
        } else {
            (7 - ep_r as usize, 7 - ep_f as usize)
        };
        data[plane_index(er, ef, 19)] = 1.0;
    }

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    Ok(PyBytes::new_bound(py, bytes).unbind())
}

/// Mirror of `chess_ai.encoding.move_to_index`. Same per-color flip as
/// `encode_board` (keeps "own side" at the bottom of the planes).
#[pyfunction]
fn move_to_index_fast(
    from_r: i32,
    from_f: i32,
    to_r: i32,
    to_f: i32,
    is_white: bool,
) -> i32 {
    let (fr, ff) = if is_white {
        (from_r, from_f)
    } else {
        (7 - from_r, 7 - from_f)
    };
    let (tr, tf) = if is_white {
        (to_r, to_f)
    } else {
        (7 - to_r, 7 - to_f)
    };
    (fr * 8 + ff) * 64 + (tr * 8 + tf)
}

#[pymodule]
fn chess_ai_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(is_in_check_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_square_attacked_by_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_move_full, m)?)?;
    m.add_function(wrap_pyfunction!(generate_children, m)?)?;
    m.add_function(wrap_pyfunction!(encode_board, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_index_fast, m)?)?;
    mcts::register(m)?;
    Ok(())
}
