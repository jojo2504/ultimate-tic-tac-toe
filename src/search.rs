use std::collections::HashMap;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    core::{Result, TicTacToe},
    movegen::generate_moves,
    network::{DualAccumulator, Network},
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

#[derive(Clone)]
pub struct Search {
    tt: HashMap<u128, TTEntry>,
    /// External accumulator stack for caller-side incremental updates
    /// (used in tournament() and similar drivers).
    pub acc: [DualAccumulator; 81],
}

impl Search {
    pub fn new() -> Self {
        Self {
            tt: HashMap::new(),
            acc: [DualAccumulator::default(); 81],
        }
    }

    fn negamax(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        mut alpha: f32,
        beta: f32,
        net: &Network,
        dual_acc: DualAccumulator,
    ) -> f32 {
        let alpha_orig = alpha;

        // Transposition table lookup
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

        // Terminal
        if board.is_game_over() {
            return match board.result() {
                Result::Win => 0.0 - 0.0001 * ((81 - board.ply) as f32), // opponent (last mover) won → current player lost
                Result::Loss => 1.0 + 0.0001 * ((81 - board.ply) as f32), // unreachable at terminal, but consistent
                Result::Draw => 0.5,
            };
        }

        // Leaf evaluation — FIX B: use the correct perspective half
        if depth == 0 {
            return net.forward(dual_acc.stm(board.turn));
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut moves = generate_moves(board);

        while moves != 0 {
            let mv: u8 = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            let mut child = board.clone();
            let delta = child.make(mv);

            let mut child_acc = dual_acc;
            child_acc.apply_delta(net, &delta);

            let score =
                1.0 - self.negamax(&child, depth - 1, 1.0 - beta, 1.0 - alpha, net, child_acc);

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
        let root_acc = DualAccumulator::new(net, board);
        let mut moves = generate_moves(board);
        let move_bit: Vec<u8> = {
            let mut temp = Vec::new();
            while moves != 0 {
                let mv: u8 = moves.trailing_zeros() as u8;
                temp.push(mv);
                moves &= moves - 1;
            }
            temp
        };

        let (_, best_mv) = move_bit
            .par_iter()
            .map(|&mv| {
                let mut child = board.clone();
                let delta = child.make(mv);

                let mut child_acc = root_acc;
                child_acc.apply_delta(net, &delta);

                let mut local_self = self.clone();
                let score =
                    1.0 - local_self.negamax(&child, depth - 1, -10.0, 10.0, net, child_acc);
                (score, mv)
            })
            .reduce(
                || (f32::NEG_INFINITY, 0),
                |(best_score, best_mv), (score, mv)| {
                    if score > best_score {
                        (score, mv)
                    } else {
                        (best_score, best_mv)
                    }
                },
            );

        best_mv
    }

    pub fn think_training(&mut self, board: &TicTacToe, depth: i32, net: &Network) -> u8 {
        let temperature = if board.ply < 6 {
            0.5
        } else if board.ply < 15 {
            0.2
        } else {
            0.05
        };
        self.think_with_noise(board, depth, net, temperature)
    }

    pub fn think_training_scored(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
    ) -> (u8, f32) {
        let temperature = if board.ply < 6 {
            0.5
        } else if board.ply < 15 {
            0.2
        } else {
            0.05
        };
        self.think_with_noise_scored(board, depth, net, temperature)
    }

    pub fn think_with_noise(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
        temperature: f32,
    ) -> u8 {
        self.think_with_noise_scored(board, depth, net, temperature)
            .0
    }

    fn think_with_noise_scored(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
        temperature: f32,
    ) -> (u8, f32) {
        let root_acc = DualAccumulator::new(net, board);

        let mut moves = generate_moves(board);
        let mut move_scores = [(0u8, 0f32); 81];
        let mut count = 0;

        while moves != 0 {
            let mv = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            let mut child = board.clone();
            let delta = child.make(mv);

            let mut child_acc = root_acc;
            child_acc.apply_delta(net, &delta);

            let score = 1.0 - self.negamax(&child, depth - 1, -10.0, 10.0, net, child_acc);
            move_scores[count] = (mv, score);
            count += 1;
        }

        let best_score = move_scores[..count]
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);

        if temperature == 0.0 {
            let best_mv = move_scores[..count]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            return (best_mv, best_score);
        }

        // Softmax temperature sampling
        let weights: Vec<f32> = move_scores[..count]
            .iter()
            .map(|(_, s)| ((s - best_score) / temperature).exp())
            .collect();

        let total: f32 = weights.iter().sum();
        let mut rng_val = rand::random::<f32>() * total;

        for (i, w) in weights.iter().enumerate() {
            rng_val -= w;
            if rng_val <= 0.0 {
                return (move_scores[i].0, best_score);
            }
        }

        (move_scores[..count].last().unwrap().0, best_score)
    }
}
