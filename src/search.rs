use std::collections::HashMap;

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

pub struct Search {
    pub tt: HashMap<u128, TTEntry>, // zobrist_key, TTEntry
    positions: [TicTacToe; 81],
    acc: [AccumulatorPair; 81],
}

impl Search {
    pub fn new() -> Self {
        Self {
            tt: HashMap::default(),
            positions: [TicTacToe::default(); 81],
            acc: [AccumulatorPair::default(); 81],
        }
    }

    fn negamax(
        &mut self,
        board: &TicTacToe,
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
            let acc = AccumulatorPair::new(net, board);
            return net.forward(acc); // [0.0, 1.0] from current player's perspective
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut moves = generate_moves(board);

        while moves != 0 {
            let mv: u8 = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            let mut child = board.clone();
            child.make(mv);

            let score = 1.0 - self.negamax(&child, depth - 1, 1.0 - beta, 1.0 - alpha, net);

            if score > best_score {
                best_score = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
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

    pub fn think(&mut self, board: &TicTacToe, depth: i32, net: &Network) -> u8 {
        // sync the positions with the ply given by the board passed in parameters
        // and clone the board at his right position in the stack
        self.positions[board.ply] = board.clone();

        let mut moves = generate_moves(board);
        let mut best_mv = moves.trailing_zeros() as u8;
        let mut best_score = f32::NEG_INFINITY;

        while moves != 0 {
            let mv: u8 = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            let mut child = board.clone();
            child.make(mv);

            let score = 1.0 - self.negamax(&child, depth - 1, 0.0, 1.0, net);
            // println!("score {}", score);
            if score > best_score {
                best_score = score;
                best_mv = mv;
            }
        }

        best_mv as u8
    }

    pub fn think_training(&mut self, board: &TicTacToe, depth: i32, net: &Network) -> u8 {
        let temperature = if board.ply < 10 {
            1.0 // opening: explore freely
        } else if board.ply < 25 {
            0.5 // midgame: some noise
        } else {
            0.0 // endgame: play best move
        };

        self.think_with_noise(board, depth, net, temperature)
    }

    fn think_with_noise(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
        temperature: f32, // 0.0 = deterministic, 1.0 = proportional, >1.0 = more random
    ) -> u8 {
        let mut moves = generate_moves(board);
        let mut move_scores: Vec<(u8, f32)> = Vec::new();

        while moves != 0 {
            let mv = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            let mut child = board.clone();
            child.make(mv);

            let score = 1.0 - self.negamax(&child, depth - 1, 0.0, 1.0, net);
            move_scores.push((mv, score));
        }

        if temperature == 0.0 {
            // deterministic — best move
            return move_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
        }

        // softmax sampling
        let max_score = move_scores
            .iter()
            .map(|(_, s)| s)
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = move_scores
            .iter()
            .map(|(_, s)| ((s - max_score) / temperature).exp())
            .collect();

        let total: f32 = weights.iter().sum();
        let mut rng_val = rand::random::<f32>() * total;

        for (i, w) in weights.iter().enumerate() {
            rng_val -= w;
            if rng_val <= 0.0 {
                return move_scores[i].0;
            }
        }

        move_scores.last().unwrap().0
    }
}
