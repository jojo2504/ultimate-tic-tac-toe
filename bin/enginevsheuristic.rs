// bootstrap.rs
//
// Generates gen0 training data using a proper alpha-beta minimax heuristic
// instead of random moves. This solves the cold-start problem: the network
// learns from competent play from day one rather than from random noise.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use ultimate_tic_tac_toe::core::TicTacToe;
use ultimate_tic_tac_toe::movegen::generate_moves;
use ultimate_tic_tac_toe::train::{Sample, flush_samples};

// ─── Heuristic parameters ----

#[derive(Clone, Copy)]
pub struct HeuristicParams {
    pub macro_win: i32,
    pub macro_two: i32,
    pub macro_one: i32,
    pub macro_fork: i32,
    pub macro_cell_weights: [i32; 9],
    pub center_macro_mult: f32,
    pub micro_two: i32,
    pub micro_one: i32,
    pub micro_center: i32,
    pub forced_board_threat: i32,
    pub mobility: i32,
}

impl Default for HeuristicParams {
    fn default() -> Self {
        Self {
            macro_win: 1_060,
            macro_two: 260,
            macro_one: 25,
            macro_fork: 420,
            macro_cell_weights: [34, 22, 34, 22, 48, 22, 34, 22, 34],
            center_macro_mult: 1.224_842_4,
            micro_two: 16,
            micro_one: 1,
            micro_center: 2,
            forced_board_threat: 500,
            mobility: 2,
        }
    }
}

// Win-score used internally by the minimax. Never leaks into Sample.search_score.
const WIN_SCORE: i32 = 1_000_000;

// Move ordering priorities (center > corner > edge, mirroring the other project)
const MOVE_PRIORITY: [i32; 9] = [2, 1, 2, 1, 3, 1, 2, 1, 2];

#[inline]
fn legal_moves(game: &TicTacToe) -> Vec<u8> {
    let moves = generate_moves(game);
    let mut move_list = Vec::with_capacity(9);
    let mut remaining = moves;
    while remaining != 0 {
        let mv = remaining.trailing_zeros() as u8;
        move_list.push(mv);
        remaining &= remaining - 1;
    }
    move_list
}

#[inline]
fn apply(game: &TicTacToe, mv: u8) -> TicTacToe {
    let mut next = game.clone();
    next.make(mv);
    next
}

#[inline]
fn is_terminal(game: &TicTacToe) -> bool {
    game.check_win() || game.is_full()
}

/// 0 = the player who moves first (Player X), 1 = Player O.
/// If your game tracks "current player" differently, adapt here.
#[inline]
fn current_player(game: &TicTacToe) -> usize {
    (game.ply as usize) % 2
}

// ─── Heuristic evaluation (ported from the other project's Board::evaluate) ──
#[derive(Clone, Copy, PartialEq)]
enum Cell {
    Empty,
    X,
    O,
}

/// Read the 9-cell local board `board_idx` from your game.
fn read_local_board(game: &TicTacToe, board_idx: usize) -> [Cell; 9] {
    let mut cells = [Cell::Empty; 9];
    let base = ultimate_tic_tac_toe::constants::MAP[board_idx] as usize;
    let indices = [
        base,
        base + 1,
        base + 2,
        base + 9,
        base + 10,
        base + 11,
        base + 18,
        base + 19,
        base + 20,
    ];

    let (cross_bb, circle_bb) = if game.turn == ultimate_tic_tac_toe::core::Symbol::Cross {
        (game.side_bitboard ^ game.bitboard, game.side_bitboard)
    } else {
        (game.side_bitboard, game.side_bitboard ^ game.bitboard)
    };

    for (i, &idx) in indices.iter().enumerate() {
        if (cross_bb >> idx) & 1 != 0 {
            cells[i] = Cell::X;
        } else if (circle_bb >> idx) & 1 != 0 {
            cells[i] = Cell::O;
        }
    }
    cells
}

/// Returns MacroCell state for each of the 9 macro-boards.
/// 0 = empty, 1 = X won, 2 = O won, 3 = draw.
fn read_macro_board(game: &TicTacToe) -> [u8; 9] {
    let mut macros = [0u8; 9];

    // cross_clear/circle_clear logic based on game.turn
    let (cross_clear, circle_clear) = if game.turn == ultimate_tic_tac_toe::core::Symbol::Cross {
        (game.side_clear ^ game.all_clear as u16, game.side_clear)
    } else {
        (game.side_clear, game.side_clear ^ game.all_clear as u16)
    };

    for i in 0..9 {
        if (cross_clear >> i) & 1 != 0 {
            macros[i] = 1;
        } else if (circle_clear >> i) & 1 != 0 {
            macros[i] = 2;
        } else if (game.all_clear >> i) & 1 != 0 {
            macros[i] = 3;
        }
    }
    macros
}

const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8], // rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8], // cols
    [0, 4, 8],
    [2, 4, 6], // diags
];

fn count_line(cells: &[Cell; 9], line: &[usize; 3]) -> (i32, i32) {
    let mut x = 0i32;
    let mut o = 0i32;
    for &i in line {
        match cells[i] {
            Cell::X => x += 1,
            Cell::O => o += 1,
            Cell::Empty => {}
        }
    }
    (x, o)
}

fn score_local_board(cells: &[Cell; 9], params: &HeuristicParams) -> i32 {
    let mut score = 0;
    for line in &WIN_LINES {
        let (x, o) = count_line(cells, line);
        if x > 0 && o > 0 {
            continue;
        }
        if x == 2 {
            score += params.micro_two;
        }
        if x == 1 {
            score += params.micro_one;
        }
        if o == 2 {
            score -= params.micro_two;
        }
        if o == 1 {
            score -= params.micro_one;
        }
    }
    // center bonus
    match cells[4] {
        Cell::X => score += params.micro_center,
        Cell::O => score -= params.micro_center,
        _ => {}
    }
    score
}

fn count_macro_line(macros: &[u8; 9], line: &[usize; 3]) -> (i32, i32) {
    let mut x = 0i32;
    let mut o = 0i32;
    for &i in line {
        match macros[i] {
            1 => x += 1,
            2 => o += 1,
            _ => {}
        }
    }
    (x, o)
}

fn count_macro_forks(macros: &[u8; 9], player_val: u8) -> i32 {
    let mut forks = 0;
    for line1 in 0..WIN_LINES.len() {
        for line2 in (line1 + 1)..WIN_LINES.len() {
            let l1 = &WIN_LINES[line1];
            let l2 = &WIN_LINES[line2];
            // lines share exactly one cell
            let shared: Vec<usize> = l1.iter().filter(|&&c| l2.contains(&c)).copied().collect();
            if shared.len() != 1 {
                continue;
            }
            let count1 = l1.iter().filter(|&&i| macros[i] == player_val).count();
            let count2 = l2.iter().filter(|&&i| macros[i] == player_val).count();
            if count1 == 1 && count2 == 1 {
                forks += 1;
            }
        }
    }
    forks
}

/// Full board evaluation, relative to player X (positive = X is winning).
pub fn evaluate_heuristic(game: &TicTacToe, params: &HeuristicParams) -> i32 {
    let macros = read_macro_board(game);

    // ── Terminal check ─────────────────────────────────────────────────────
    for line in &WIN_LINES {
        let (x, o) = count_macro_line(&macros, line);
        if x == 3 {
            return WIN_SCORE;
        }
        if o == 3 {
            return -WIN_SCORE;
        }
    }
    if macros.iter().all(|&m| m != 0) {
        let x_boards = macros.iter().filter(|&&m| m == 1).count() as i32;
        let o_boards = macros.iter().filter(|&&m| m == 2).count() as i32;
        return match x_boards.cmp(&o_boards) {
            std::cmp::Ordering::Greater => WIN_SCORE,
            std::cmp::Ordering::Less => -WIN_SCORE,
            std::cmp::Ordering::Equal => 0,
        };
    }

    // ── Positional score ───────────────────────────────────────────────────
    let mut score = 0i32;

    // Macro-line threats
    for line in &WIN_LINES {
        let (x, o) = count_macro_line(&macros, line);
        if x > 0 && o > 0 {
            continue;
        }
        if x == 2 {
            score += params.macro_two;
        }
        if x == 1 {
            score += params.macro_one;
        }
        if o == 2 {
            score -= params.macro_two;
        }
        if o == 1 {
            score -= params.macro_one;
        }
    }

    // Fork bonus
    score += count_macro_forks(&macros, 1) * params.macro_fork;
    score -= count_macro_forks(&macros, 2) * params.macro_fork;

    // Per-macro-board contributions
    for i in 0..9usize {
        let mult = if i == 4 {
            params.center_macro_mult
        } else {
            1.0
        };
        match macros[i] {
            1 => score += ((params.macro_win + params.macro_cell_weights[i]) as f32 * mult) as i32,
            2 => score -= ((params.macro_win + params.macro_cell_weights[i]) as f32 * mult) as i32,
            0 => {
                let cells = read_local_board(game, i);
                let local = score_local_board(&cells, params);
                score += (((local + params.macro_cell_weights[i] / 3) as f32) * mult) as i32;
            }
            _ => {} // draw
        }
    }

    score
}

// ─── Alpha-beta minimax (negamax form) ────────────────────────────────────────

const MAX_DEPTH: u32 = 5; // bootstrap depth — higher = better data, slower generation

fn negamax(
    game: &TicTacToe,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    params: &HeuristicParams,
) -> i32 {
    if is_terminal(game) || depth == 0 {
        let raw = evaluate_heuristic(game, params);
        return if current_player(game) == 0 { raw } else { -raw };
    }

    let mut moves = legal_moves(game);
    if moves.is_empty() {
        let raw = evaluate_heuristic(game, params);
        return if current_player(game) == 0 { raw } else { -raw };
    }

    moves.sort_by_key(|&mv| {
        let board_idx =
            ultimate_tic_tac_toe::constants::CELL_TO_SUBBOARD_INDEX[mv as usize] as usize;
        let base = ultimate_tic_tac_toe::constants::MAP[board_idx] as usize;
        let diff = mv as usize - base;
        let micro_idx = match diff {
            0 => 0,
            1 => 1,
            2 => 2,
            9 => 3,
            10 => 4,
            11 => 5,
            18 => 6,
            19 => 7,
            20 => 8,
            _ => unreachable!(),
        };
        -MOVE_PRIORITY[micro_idx]
    });

    let mut best = i32::MIN + 1;
    for mv in moves {
        let child = apply(game, mv);
        // Negate because negamax alternates perspective
        let score = -negamax(&child, depth - 1, -beta, -alpha, params);
        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        } // beta cut-off
    }
    best
}

/// Pick the best move using alpha-beta negamax at the configured depth.
fn best_move(game: &TicTacToe, params: &HeuristicParams) -> Option<u8> {
    let moves = legal_moves(game);
    if moves.is_empty() {
        return None;
    }

    let mut ordered = moves.clone();
    ordered.sort_by_key(|&mv| {
        let board_idx =
            ultimate_tic_tac_toe::constants::CELL_TO_SUBBOARD_INDEX[mv as usize] as usize;
        let base = ultimate_tic_tac_toe::constants::MAP[board_idx] as usize;
        let diff = mv as usize - base;
        let micro_idx = match diff {
            0 => 0,
            1 => 1,
            2 => 2,
            9 => 3,
            10 => 4,
            11 => 5,
            18 => 6,
            19 => 7,
            20 => 8,
            _ => unreachable!(),
        };
        -MOVE_PRIORITY[micro_idx]
    });

    let mut best_mv = ordered[0];
    let mut best_score = i32::MIN + 1;
    let mut alpha = i32::MIN + 1;
    let beta = i32::MAX;

    for mv in ordered {
        let child = apply(game, mv);
        let score = -negamax(&child, MAX_DEPTH - 1, -beta, -alpha, params);
        if score > best_score {
            best_score = score;
            best_mv = mv;
        }
        if score > alpha {
            alpha = score;
        }
    }

    Some(best_mv)
}

// ─── Map raw heuristic score → [0, 1] for Sample.search_score ────────────────
fn score_to_search_score(raw: i32, is_x_to_move: bool) -> f32 {
    // Clamp to [-WIN_SCORE, WIN_SCORE] then shift to [0, 2*WIN_SCORE] then normalise.
    let clamped = raw.clamp(-WIN_SCORE, WIN_SCORE);
    // Flip perspective if it's O's turn (score was computed for X).
    let from_current = if is_x_to_move { clamped } else { -clamped };
    (from_current + WIN_SCORE) as f32 / (2 * WIN_SCORE) as f32
}

// ─── Single game generation ────────────────────────────────────────────────────

use ultimate_tic_tac_toe::network::Network;
use ultimate_tic_tac_toe::search::Search;

fn play_game(params: &HeuristicParams, net: &Network, depth: i32, heuristic_player: usize) {
    let mut board = TicTacToe::new();
    let mut search = Search::new();

    println!("Starting game. Heuristic is player {}", heuristic_player);

    while !is_terminal(&board) {
        if current_player(&board) == heuristic_player {
            let mv = best_move(&board, params).expect("Heuristic found no move");
            println!("Heuristic plays: {} or ({}, {})", mv, mv / 9, mv % 9);
            board.make(mv);
        } else {
            let mv = search.think(&board, depth, net, None);
            println!("Engine plays: {} or ({}, {})", mv, mv / 9, mv % 9);
            board.make(mv as u8);
        }
        println!("{}", board);
    }

    if board.check_win() {
        let winner = board.turn.swap();
        let winner_idx = if winner == ultimate_tic_tac_toe::core::Symbol::Cross {
            0
        } else {
            1
        };
        if winner_idx == heuristic_player {
            println!("Result: Heuristic wins!");
        } else {
            println!("Result: Engine wins!");
        }
    } else {
        println!("Result: Draw!");
    }
}

pub fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <generation_number> <depth>", args[0]);
        std::process::exit(1);
    }

    let gen_num = &args[1];
    let depth: i32 = args[2].parse().unwrap();
    let net = Network::load(format!("databin/gen{}_weights.bin", gen_num));
    let params = HeuristicParams::default();

    println!("=== GAME 1: Heuristic (X) vs Engine (O) ===");
    play_game(&params, &net, depth, 0);

    println!("\n=== GAME 2: Engine (X) vs Heuristic (O) ===");
    play_game(&params, &net, depth, 1);

    Ok(())
}
