use std::{
    cmp::max,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use rayon::prelude::*;

use crate::{core::TicTacToe, movegen::generate_moves};

#[derive(Default, Clone)]
enum NodeType {
    EXACT,
    LOWERBOUND,
    UPPERBOUND,
    #[default]
    None,
}

#[derive(Default, Clone)]
pub struct TTEntry {
    flag: NodeType,
    depth: i32,
    value: i32,
}

impl TTEntry {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}

pub struct Evaluation;

impl Evaluation {
    pub fn evaluate(board: &TicTacToe) -> i32 {
        todo!()
    }
}

#[derive(Default, Clone)]
pub struct Search {
    pub depth: i32,
    pub tt: HashMap<u128, TTEntry>, // zobrist_key, TTEntry
}

impl Search {
    pub fn new(depth: i32) -> Self {
        Self {
            depth: depth,
            ..Default::default()
        }
    }

    fn negamax(
        &mut self,
        board: &mut TicTacToe,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        color: i32,
    ) -> i32 {
        let alpha_orig = alpha;

        if let Some(tt_entry) = self.tt.get(&board.zobrist_key)
            && tt_entry.depth >= depth
        {
            match tt_entry.flag {
                NodeType::EXACT => return tt_entry.value,
                NodeType::LOWERBOUND if tt_entry.value >= beta => return tt_entry.value,
                NodeType::UPPERBOUND if tt_entry.value <= alpha => return tt_entry.value,
                _ => (),
            }
        }

        if depth == 0 || board.check_win() {
            return color * Evaluation::evaluate(board);
        }

        let mut child_nodes = generate_moves(board);

        let mut best_score = i32::MIN;

        while child_nodes != 0 {
            let child = 1 << child_nodes.trailing_zeros();
            child_nodes &= child_nodes - 1;

            todo!("implement copy-make");

            board.make(child);
            best_score = max(
                best_score,
                self.negamax(
                    board,
                    depth - 1,
                    beta.saturating_neg(),
                    alpha.saturating_neg(),
                    -color,
                )
                .saturating_neg(),
            );

            alpha = max(alpha, best_score);
            if alpha >= beta {
                break;
            }
        }

        let mut tt_entry = TTEntry::new();
        if best_score <= alpha_orig {
            tt_entry.flag = NodeType::UPPERBOUND;
        } else if best_score >= beta {
            tt_entry.flag = NodeType::LOWERBOUND;
        } else {
            tt_entry.flag = NodeType::EXACT;
        }

        tt_entry.depth = depth;
        tt_entry.value = best_score;
        self.tt.insert(board.zobrist_key, tt_entry);

        return best_score;
    }

    pub fn think(&mut self, game: &mut TicTacToe) -> Option<i32> {
        todo!()
    }
}
