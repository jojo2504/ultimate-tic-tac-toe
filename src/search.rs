use std::{
    cmp::max,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use rayon::prelude::*;

use crate::{
    core::{Result, TicTacToe},
    movegen::generate_moves,
    network::{AccumulatorPair, Network},
};

#[derive(Default, Clone)]
enum NodeType {
    Exact,
    LowerBound,
    UpperBound,
    #[default]
    None,
}

#[derive(Default, Clone)]
pub struct TTEntry {
    flag: NodeType,
    depth: i32,
    value: f32,
}

impl TTEntry {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub struct Search {
    pub tt: HashMap<u128, TTEntry>, // zobrist_key, TTEntry
    ply: usize,
    positions: [TicTacToe; 81],
    acc: [AccumulatorPair; 81],
}

impl Search {
    pub fn new() -> Self {
        Self {
            tt: HashMap::default(),
            ply: 0,
            positions: [TicTacToe::default(); 81],
            acc: [AccumulatorPair::default(); 81],
        }
    }

    fn negamax(
        &mut self,
        board: &mut TicTacToe,
        depth: i32,
        mut alpha: f32,
        beta: f32,
        net: &Network,
    ) -> f32 {
        let alpha_orig = alpha;

        // transposition table lookup
        if let Some(tt_entry) = self.tt.get(&board.zobrist_key) {
            if tt_entry.depth >= depth {
                match tt_entry.flag {
                    NodeType::Exact => return tt_entry.value,
                    NodeType::LowerBound => {
                        if tt_entry.value >= beta {
                            return tt_entry.value;
                        }
                    }
                    NodeType::UpperBound => {
                        if tt_entry.value <= alpha {
                            return tt_entry.value;
                        }
                    }
                    _ => (),
                }
            }
        }

        // terminal / leaf
        if board.is_game_over() {
            // game over: return exact result from current player's perspective
            return match board.result() {
                Result::Win => 1.0,  // current player won
                Result::Loss => 0.0, // current player lost
                Result::Draw => 0.5,
            };
        }

        if depth == 0 {
            return net.forward(self.acc[self.ply]); // [0.0, 1.0] from current player's perspective
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut moves = generate_moves(board);

        while moves != 0 {
            let mv = 1 << moves.trailing_zeros();
            moves &= moves - 1;

            self.positions[self.ply + 1] = self.positions[self.ply].clone();
            self.ply += 1;
            board.make(mv);
            // negamax: flip score with (1.0 - score) instead of negating
            // because our scores are in [0.0, 1.0], not [-inf, +inf]
            let score = 1.0 - self.negamax(board, depth - 1, 1.0 - beta, 1.0 - alpha, net);
            self.ply -= 1;

            if score > best_score {
                best_score = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break; // cutoff
            }
        }

        // store in transposition table
        let flag = if best_score <= alpha_orig {
            NodeType::UpperBound
        } else if best_score >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };

        self.tt.insert(
            board.zobrist_key,
            TTEntry {
                depth,
                value: best_score,
                flag,
            },
        );

        best_score
    }

    pub fn think(&mut self, board: &mut TicTacToe, depth: i32, net: &Network) -> u8 {
        let mut moves = generate_moves(board);
        let mut best_mv = 1 << moves.trailing_zeros();
        let mut best_score = f32::NEG_INFINITY;

        while moves != 0 {
            let mv = 1 << moves.trailing_zeros();
            moves &= moves - 1;

            self.positions[self.ply + 1] = self.positions[self.ply].clone();
            self.ply += 1;
            board.make(mv);
            let score = 1.0 - self.negamax(board, depth - 1, 0.0, 1.0, net);
            self.ply -= 1;

            if score > best_score {
                best_score = score;
                best_mv = mv;
            }
        }

        best_mv
    }
}
