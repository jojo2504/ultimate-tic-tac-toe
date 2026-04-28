use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::AtomicBool},
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    core::{Result, TicTacToe},
    movegen::generate_moves,
    network::{DualAccumulator, Network, get_bucket},
    train::POLICY_LEN,
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
    tt: Arc<Mutex<HashMap<u128, TTEntry>>>,
    /// External accumulator stack for caller-side incremental updates
    /// (used in tournament() and similar drivers).
    pub acc: [DualAccumulator; 81],
    /// History heuristic table: history[from_square] = score boost.
    /// Updated each time a move causes a beta-cutoff. Used to seed move
    /// ordering before the network has a usable policy head.
    history: [i32; 81],
}

impl Search {
    pub fn new() -> Self {
        Self {
            tt: Arc::new(Mutex::new(HashMap::new())),
            acc: [DualAccumulator::default(); 81],
            history: [0; 81],
        }
    }

    /// Order legal moves before searching. Two heuristics, in priority order:
    ///   1. Network policy (if available) — strongest signal once the policy
    ///      head is trained. If untrained, all logits are zero → no effect.
    ///   2. History heuristic — moves that previously caused beta-cutoffs.
    ///
    /// `moves_in` is a bitmask; `out` is a fixed-size buffer. Returns count.
    fn order_moves(
        &self,
        moves_bitmask: u128,
        net: &Network,
        dual_acc: &DualAccumulator,
        board: &TicTacToe,
        out: &mut [u8; 81],
    ) -> usize {
        let mut moves = moves_bitmask;
        let mut scored = [(0u8, 0i64); 81];
        let mut count = 0;

        // Skip the policy forward entirely when the head is untrained: it
        // would produce uniform values and just cost CPU. Once Phase 2 lands
        // and training begins, this branch flips automatically.
        let policy_trained = net.is_policy_trained();
        let policy = if policy_trained {
            net.forward_policy(dual_acc.stm(board.turn))
        } else {
            [0.0f32; 81]
        };

        while moves != 0 {
            let mv = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            // Composite score: 1000 × policy_prob + history.
            // Policy dominates when trained; history is the sole signal otherwise.
            let policy_score = (policy[mv as usize] * 1000.0) as i64;
            let hist_score = self.history[mv as usize] as i64;
            scored[count] = (mv, policy_score + hist_score);
            count += 1;
        }

        // Insertion sort, descending. n ≤ 9 so this is fine.
        for i in 1..count {
            let mut j = i;
            while j > 0 && scored[j - 1].1 < scored[j].1 {
                scored.swap(j - 1, j);
                j -= 1;
            }
        }

        for i in 0..count {
            out[i] = scored[i].0;
        }
        count
    }

    fn negamax(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        mut alpha: f32,
        beta: f32,
        net: &Network,
        dual_acc: DualAccumulator,
        stop: Option<&Arc<AtomicBool>>,
    ) -> f32 {
        if let Some(stop_signal) = stop {
            if stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                return 0.0;
            }
        }

        let alpha_orig = alpha;

        // Transposition table lookup
        if let Some(tt_entry) = self.tt.lock().unwrap().get(&board.zobrist_key) {
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
                Result::Win => 0.0 - 0.0001 * ((81 - board.ply) as f32),
                Result::Loss => 1.0 + 0.0001 * ((81 - board.ply) as f32),
                Result::Draw => 0.5,
            };
        }

        if depth == 0 {
            let bucket = get_bucket(board.ply);
            return net.forward(dual_acc.stm(board.turn), bucket);
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut ordered = [0u8; 81];
        let count = self.order_moves(generate_moves(board), net, &dual_acc, board, &mut ordered);

        // Phase 3.2: Late Move Reductions (LMR).
        //
        //   First 3 moves      → searched at full depth (depth-1 child)
        //   Move #4 and later  → searched at depth-2 (one ply shallower)
        //
        // When a reduced search beats alpha we re-search at full depth, so the
        // worst case is "did some extra cheap searches"; the best case (most
        // moves) is real Elo from focusing CPU on the moves the ordering
        // heuristics liked. The min(1) clamp guarantees we never recurse to
        // depth 0 from a reduction.
        const LMR_FULL_THRESHOLD: usize = 3;

        for i in 0..count {
            let mv = ordered[i];

            let mut child = board.clone();
            let delta = child.make(mv);

            let mut child_acc = dual_acc;
            child_acc.apply_delta(net, &delta, board, &child);

            let do_lmr = depth >= 3 && i >= LMR_FULL_THRESHOLD;
            let reduced_depth = if do_lmr {
                (depth - 2).max(1)
            } else {
                depth - 1
            };

            let mut score = 1.0
                - self.negamax(
                    &child,
                    reduced_depth,
                    1.0 - beta,
                    1.0 - alpha,
                    net,
                    child_acc,
                    stop,
                );

            // Re-search at full depth if the reduced search looked promising —
            // otherwise we might prune a real best move that got buried by
            // imperfect ordering.
            if do_lmr && score > alpha {
                score = 1.0
                    - self.negamax(
                        &child,
                        depth - 1,
                        1.0 - beta,
                        1.0 - alpha,
                        net,
                        child_acc,
                        stop,
                    );
            }

            if score > best_score {
                best_score = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                // Beta cutoff: reward this move in the history table so it's
                // tried earlier next time we see a similar position.
                self.history[mv as usize] = self.history[mv as usize].saturating_add(depth * depth);
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

        self.tt.lock().unwrap().insert(
            board.zobrist_key,
            TTEntry {
                depth,
                value: best_score,
                flag,
            },
        );

        best_score
    }

    pub fn think(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
        stop: Option<&Arc<AtomicBool>>,
    ) -> u8 {
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
                child_acc.apply_delta(net, &delta, board, &child);

                let mut local_self = self.clone();
                let score =
                    1.0 - local_self.negamax(&child, depth - 1, 0.0, 1.0, net, child_acc, stop);
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
        let temperature = pick_temperature(board.ply);
        self.think_with_noise(board, depth, net, temperature)
    }

    pub fn think_training_scored(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
    ) -> (u8, f32) {
        let temperature = pick_temperature(board.ply);
        let (mv, score, _policy) = self.think_with_noise_full(board, depth, net, temperature, true);
        (mv, score)
    }

    /// Phase 1 entry point: returns the move played, the search score (bounded
    /// to [0,1]), AND the full 81-wide policy distribution to be stored in the
    /// training Sample. `add_dirichlet=true` injects Dirichlet noise at the root
    /// to maintain exploration during self-play.
    pub fn think_training_with_policy(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
    ) -> (u8, f32, [f32; POLICY_LEN]) {
        let temperature = pick_temperature(board.ply);
        self.think_with_noise_full(board, depth, net, temperature, true)
    }

    pub fn think_with_noise(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
        temperature: f32,
    ) -> u8 {
        self.think_with_noise_full(board, depth, net, temperature, false)
            .0
    }

    /// Core self-play move chooser.
    ///
    /// 1. Searches each legal move at depth-1 with negamax.
    /// 2. Builds a softmax distribution over moves using temperature.
    /// 3. (Optional) mixes Dirichlet(α) noise into that distribution.
    /// 4. Samples a move from the noised distribution.
    /// 5. Returns (move, best_score, policy_target_for_training).
    ///
    /// The policy target is the *un-noised* softmax — we want the network to
    /// learn the actual search-score distribution, not the noise.
    fn think_with_noise_full(
        &mut self,
        board: &TicTacToe,
        depth: i32,
        net: &Network,
        temperature: f32,
        add_dirichlet: bool,
    ) -> (u8, f32, [f32; POLICY_LEN]) {
        let root_acc = DualAccumulator::new(net, board);

        // 1. Score every legal move
        let mut moves = generate_moves(board);
        let mut move_scores = [(0u8, 0f32); 81];
        let mut count = 0;

        while moves != 0 {
            let mv = moves.trailing_zeros() as u8;
            moves &= moves - 1;

            let mut child = board.clone();
            let delta = child.make(mv);

            let mut child_acc = root_acc;
            child_acc.apply_delta(net, &delta, board, &child);

            let score = 1.0 - self.negamax(&child, depth - 1, 0.0, 1.0, net, child_acc, None);
            move_scores[count] = (mv, score);
            count += 1;
        }

        let best_score = move_scores[..count]
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);

        // 2. Softmax → policy distribution (used as training target)
        let mut policy = [0.0f32; POLICY_LEN];

        if temperature == 0.0 {
            // Greedy: one-hot policy on the best move
            let best_mv = move_scores[..count]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            policy[best_mv as usize] = 1.0;
            return (best_mv, best_score, policy);
        }

        // Standard softmax (numerically stable: subtract best_score)
        let mut weights = [0.0f32; 81];
        let mut total = 0.0f32;
        for i in 0..count {
            let (mv, s) = move_scores[i];
            let w = ((s - best_score) / temperature).exp();
            weights[i] = w;
            total += w;
            policy[mv as usize] = w; // unnormalised; normalised below
        }
        if total > 0.0 {
            for i in 0..POLICY_LEN {
                policy[i] /= total;
            }
            for i in 0..count {
                weights[i] /= total;
            }
        }

        // 3. Optional Dirichlet(α) mixing for exploration at the root.
        //    Only the SAMPLING distribution is noised; the policy target
        //    stored for training stays clean.
        let sampling_weights: [f32; 81] = if add_dirichlet {
            mix_dirichlet_at_root(&policy, count, &move_scores)
        } else {
            let mut w = [0.0f32; 81];
            for i in 0..count {
                w[move_scores[i].0 as usize] = weights[i];
            }
            w
        };

        // 4. Sample from sampling_weights
        let sample_total: f32 = sampling_weights.iter().sum();
        let mut rng_val = rand::random::<f32>() * sample_total;
        let mut chosen = move_scores[count - 1].0;
        for i in 0..count {
            let mv = move_scores[i].0;
            rng_val -= sampling_weights[mv as usize];
            if rng_val <= 0.0 {
                chosen = mv;
                break;
            }
        }

        (chosen, best_score, policy)
    }

    /// Iterative deepening kept for time-limited play.
    pub fn iterative_deepening(&mut self, board: &TicTacToe, net: &Network, stop: Arc<AtomicBool>) {
        let mut current_depth = 1;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            self.think(board, current_depth, net, Some(&stop.clone()));
            current_depth += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dirichlet noise (AlphaZero-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Dirichlet concentration. Lower α = more peaky noise. AlphaZero uses 0.3
/// for chess (35-move avg branching). Ultimate TTT has ~5-9 legal moves so
/// we use a higher α to spread noise more evenly.
const DIRICHLET_ALPHA: f32 = 1.5;

/// Fraction of noise to mix in: 25% noise, 75% original policy.
const DIRICHLET_EPSILON: f32 = 0.25;

/// Mix Dirichlet(α) noise into the root policy. Only legal-move slots receive
/// noise; illegal slots remain 0. Returns an 81-wide weight array suitable
/// for sampling.
///
/// Noise generation uses the Gamma(α, 1) trick: x_i ~ Gamma(α), then
/// d_i = x_i / Σ x_i is Dirichlet(α).
fn mix_dirichlet_at_root(
    policy: &[f32; POLICY_LEN],
    count: usize,
    move_scores: &[(u8, f32); 81],
) -> [f32; 81] {
    // Sample Dirichlet over the `count` legal moves
    let mut gammas = [0.0f32; 81];
    let mut g_sum = 0.0f32;
    for i in 0..count {
        let g = sample_gamma(DIRICHLET_ALPHA);
        gammas[i] = g;
        g_sum += g;
    }
    if g_sum <= 0.0 {
        // Degenerate case — fall back to uniform-on-legals
        let uniform = 1.0 / count as f32;
        let mut out = [0.0f32; 81];
        for i in 0..count {
            out[move_scores[i].0 as usize] = uniform;
        }
        return out;
    }

    let mut out = [0.0f32; 81];
    for i in 0..count {
        let mv = move_scores[i].0 as usize;
        let dirichlet_i = gammas[i] / g_sum;
        out[mv] = (1.0 - DIRICHLET_EPSILON) * policy[mv] + DIRICHLET_EPSILON * dirichlet_i;
    }
    out
}

/// Marsaglia & Tsang method for Gamma(α, 1). Works for α > 0.
/// For α < 1 we use the boost-and-rescale trick: G(α) = G(α+1) × U^(1/α).
fn sample_gamma(alpha: f32) -> f32 {
    if alpha < 1.0 {
        let g = sample_gamma_marsaglia(alpha + 1.0);
        let u: f32 = rand::random();
        return g * u.powf(1.0 / alpha);
    }
    sample_gamma_marsaglia(alpha)
}

fn sample_gamma_marsaglia(alpha: f32) -> f32 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let z = standard_normal();
            let v = (1.0 + c * z).powi(3);
            if v > 0.0 {
                break (z, v);
            }
        };
        let u: f32 = rand::random();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Box-Muller standard normal (mean 0, std 1).
fn standard_normal() -> f32 {
    let u1: f32 = rand::random::<f32>().max(1e-10);
    let u2: f32 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperature schedule
// ─────────────────────────────────────────────────────────────────────────────

fn pick_temperature(ply: usize) -> f32 {
    if ply < 6 {
        0.5
    } else if ply < 15 {
        0.2
    } else {
        0.05
    }
}