use bytemuck::{Pod, Zeroable, checked::try_from_bytes};
use std::fs;

use crate::{
    constants::FEATURES_COUNT,
    core::{MoveDelta, Symbol, TicTacToe},
};

// ─────────────────────────────────────────────────────────────────────────────
// Network weights
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Network {
    pub w0: [[f32; FEATURES_COUNT]; 128],
    pub b0: [f32; 128],
    pub w1: [[f32; 128]; 64],
    pub b1: [f32; 64],
    pub w2: [[f32; 64]; 1],
    pub b2: [f32; 1],
}

#[inline(always)]
fn screlu(x: f32) -> f32 {
    let clamped = x.clamp(0.0, 1.0);
    clamped * clamped
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Network {
    pub fn load(path: String) -> Box<Self> {
        let bytes = fs::read(path).expect("failed to read weights file");
        Box::new(*try_from_bytes::<Network>(&bytes).expect(&format!(
            "invalid alignment or size:\nsize of bytes: {}\n",
            std::mem::size_of_val(&bytes)
        )))
    }

    /// Forward pass.  `acc` is a *single-perspective* hidden layer – i.e. the
    /// caller must already have selected the STM half of a DualAccumulator.
    pub fn forward(&self, acc: &[f32; 128]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.forward_avx2(acc) };
            }
        }
        self.forward_scalar(acc)
    }

    fn forward_scalar(&self, acc: &[f32; 128]) -> f32 {
        let mut screlu_input = [0f32; 128];
        for j in 0..128 {
            screlu_input[j] = screlu(acc[j]);
        }

        let mut h1 = [0f32; 64];
        for i in 0..64 {
            let mut sum = self.b1[i];
            let w = &self.w1[i];
            for j in 0..128 {
                sum += w[j] * screlu_input[j];
            }
            h1[i] = screlu(sum);
        }

        let mut out = self.b2[0];
        for i in 0..64 {
            out += self.w2[0][i] * h1[i];
        }
        sigmoid(out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, acc: &[f32; 128]) -> f32 {
        use std::arch::x86_64::*;

        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);

        let mut screlu_buf = [0f32; 128];
        let stm_ptr = acc.as_ptr();
        let out_ptr = screlu_buf.as_mut_ptr();
        for k in (0..128).step_by(8) {
            let x = _mm256_loadu_ps(stm_ptr.add(k));
            let clamped = _mm256_min_ps(_mm256_max_ps(x, zero), one);
            _mm256_storeu_ps(out_ptr.add(k), _mm256_mul_ps(clamped, clamped));
        }

        let mut h1 = [0f32; 64];
        let s_ptr = screlu_buf.as_ptr();

        for i in 0..64 {
            let w_ptr = self.w1[i].as_ptr();
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            let mut j = 0;
            while j < 128 {
                a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w_ptr.add(j)),
                    _mm256_loadu_ps(s_ptr.add(j)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w_ptr.add(j + 8)),
                    _mm256_loadu_ps(s_ptr.add(j + 8)),
                    a1,
                );
                a2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w_ptr.add(j + 16)),
                    _mm256_loadu_ps(s_ptr.add(j + 16)),
                    a2,
                );
                a3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w_ptr.add(j + 24)),
                    _mm256_loadu_ps(s_ptr.add(j + 24)),
                    a3,
                );
                j += 32;
            }
            a0 = _mm256_add_ps(a0, a1);
            a2 = _mm256_add_ps(a2, a3);
            a0 = _mm256_add_ps(a0, a2);
            let hi = _mm256_extractf128_ps(a0, 1);
            let lo = _mm256_castps256_ps128(a0);
            let s128 = _mm_add_ps(lo, hi);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
            let dot = _mm_cvtss_f32(s32) + self.b1[i];
            let c = dot.clamp(0.0, 1.0);
            h1[i] = c * c;
        }

        let h_ptr = h1.as_ptr();
        let w2_ptr = self.w2[0].as_ptr();
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut j = 0;
        while j < 64 {
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w2_ptr.add(j)),
                _mm256_loadu_ps(h_ptr.add(j)),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(w2_ptr.add(j + 8)),
                _mm256_loadu_ps(h_ptr.add(j + 8)),
                a1,
            );
            j += 16;
        }
        a0 = _mm256_add_ps(a0, a1);
        let hi = _mm256_extractf128_ps(a0, 1);
        let lo = _mm256_castps256_ps128(a0);
        let s128 = _mm_add_ps(lo, hi);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
        sigmoid(_mm_cvtss_f32(s32) + self.b2[0])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn add_feature(acc: &mut [f32; 128], net: &Network, f: usize) {
    for i in 0..128 {
        acc[i] += net.w0[i][f];
    }
}

#[inline(always)]
fn sub_feature(acc: &mut [f32; 128], net: &Network, f: usize) {
    for i in 0..128 {
        acc[i] -= net.w0[i][f];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DualAccumulator
// ─────────────────────────────────────────────────────────────────────────────

/// FIX B: Dual-perspective NNUE accumulator (the "Stockfish pattern").
///
/// `acc[0]` is kept from Cross's point of view (Cross = side to move).
/// `acc[1]` is kept from Circle's point of view (Circle = side to move).
///
/// When a piece is played by Cross:
///   – acc[0] sees it as a "my piece"      → STM   feature [sq]
///   – acc[1] sees it as an "opponent piece" → NSTM feature [81 + sq]
///
/// When a piece is played by Circle:
///   – acc[1] sees it as a "my piece"      → STM   feature [sq]
///   – acc[0] sees it as an "opponent piece" → NSTM feature [81 + sq]
///
/// Focus and all_clear features are symmetric: updated identically in both.
///
/// At eval time, `stm()` returns the accumulator half for whichever side is
/// to move, so the network always receives a consistent STM-relative input.
#[derive(Debug, Clone, Copy)]
pub struct DualAccumulator {
    /// [0] = Cross-as-STM half, [1] = Circle-as-STM half
    pub acc: [[f32; 128]; 2],
}

impl DualAccumulator {
    /// Build both halves from scratch for `board`'s current position.
    pub fn new(net: &Network, board: &TicTacToe) -> Self {
        // Recover absolute (physical) piece ownership
        let (cross_bb, circle_bb, cross_clear, circle_clear) = match board.turn as i32 {
            1 => (
                board.side_bitboard ^ board.bitboard, // Cross STM → STM pieces = Cross
                board.side_bitboard,
                board.side_clear ^ board.all_clear as u16,
                board.side_clear,
            ),
            -1 => (
                board.side_bitboard,
                board.side_bitboard ^ board.bitboard, // Circle STM → STM pieces = Circle
                board.side_clear,
                board.side_clear ^ board.all_clear as u16,
            ),
            _ => unreachable!(),
        };

        let mut da = DualAccumulator { acc: [net.b0; 2] };

        // Build acc[0]  (Cross = my, Circle = opp)
        // Build acc[1]  (Circle = my, Cross = opp)
        for i in 0..81 {
            if (cross_bb >> i) & 1 != 0 {
                add_feature(&mut da.acc[0], net, i); // Cross: my piece in acc[0]
                add_feature(&mut da.acc[1], net, 81 + i); // Cross: opp piece in acc[1]
            }
            if (circle_bb >> i) & 1 != 0 {
                add_feature(&mut da.acc[0], net, 81 + i); // Circle: opp piece in acc[0]
                add_feature(&mut da.acc[1], net, i); // Circle: my piece in acc[1]
            }
        }

        // Cleared meta-board (Cross)
        for i in 0..9 {
            if (cross_clear >> i) & 1 != 0 {
                add_feature(&mut da.acc[0], net, 162 + i); // my clear for acc[0]
                add_feature(&mut da.acc[1], net, 171 + i); // opp clear for acc[1]
            }
        }
        // Cleared meta-board (Circle)
        for i in 0..9 {
            if (circle_clear >> i) & 1 != 0 {
                add_feature(&mut da.acc[0], net, 171 + i); // opp clear for acc[0]
                add_feature(&mut da.acc[1], net, 162 + i); // my clear for acc[1]
            }
        }

        // all_clear (symmetric)
        for i in 0..9 {
            if (board.all_clear >> i) & 1 != 0 {
                add_feature(&mut da.acc[0], net, 180 + i);
                add_feature(&mut da.acc[1], net, 180 + i);
            }
        }

        // Focus (symmetric)
        let focus_feat = match board.current_focus {
            Some(f) => 189 + f as usize,
            None => 198,
        };
        add_feature(&mut da.acc[0], net, focus_feat);
        add_feature(&mut da.acc[1], net, focus_feat);

        da
    }

    /// Return the accumulator half for whichever side is to move.
    pub fn stm(&self, turn: Symbol) -> &[f32; 128] {
        match turn {
            Symbol::Cross => &self.acc[0],
            Symbol::Circle => &self.acc[1],
        }
    }

    /// Incrementally update both halves after a move described by `delta`.
    ///
    /// FIX B fixes three bugs from the original single-accumulator design:
    ///   1. Perspective-flip: each half updates with the correct STM/NSTM sign.
    ///   2. Subtraction: old_focus is properly removed before new_focus is added.
    ///   3. Double trailing_zeros: cleared_board is already an index 0-8,
    ///      not a bitboard, so we use it directly without trailing_zeros().
    pub fn apply_delta(&mut self, net: &Network, delta: &MoveDelta) {
        let sq = delta.square as usize;

        // ── 1. Piece placement ────────────────────────────────────────────────
        match delta.turn {
            Symbol::Cross => {
                // Cross's piece: "my piece" (STM feature 0..81) in acc[0],
                //                "opp piece" (NSTM feature 81..162) in acc[1]
                add_feature(&mut self.acc[0], net, sq);
                add_feature(&mut self.acc[1], net, 81 + sq);
            }
            Symbol::Circle => {
                // Circle's piece: "my piece" in acc[1], "opp piece" in acc[0]
                add_feature(&mut self.acc[1], net, sq);
                add_feature(&mut self.acc[0], net, 81 + sq);
            }
        }

        // ── 2. Cleared sub-board ─────────────────────────────────────────────
        if let Some(b) = delta.cleared_board {
            let b = b as usize;
            // FIX B-3: b is already an index (0-8); do NOT call trailing_zeros() on it.
            match delta.turn {
                Symbol::Cross => {
                    // Cross cleared it: STM-clear (162+b) for acc[0], NSTM-clear (171+b) for acc[1]
                    add_feature(&mut self.acc[0], net, 162 + b);
                    add_feature(&mut self.acc[1], net, 171 + b);
                }
                Symbol::Circle => {
                    add_feature(&mut self.acc[1], net, 162 + b);
                    add_feature(&mut self.acc[0], net, 171 + b);
                }
            }
            // all_clear is symmetric
            add_feature(&mut self.acc[0], net, 180 + b);
            add_feature(&mut self.acc[1], net, 180 + b);
        }

        // ── 3. Focus change (symmetric across both halves) ───────────────────
        // FIX B-2: subtract the old focus BEFORE adding the new one.
        let old_feat = match delta.old_focus {
            Some(f) => 189 + f as usize,
            None => 198, // free-choice flag
        };
        let new_feat = match delta.new_focus {
            Some(f) => 189 + f as usize,
            None => 198,
        };
        sub_feature(&mut self.acc[0], net, old_feat);
        sub_feature(&mut self.acc[1], net, old_feat);
        add_feature(&mut self.acc[0], net, new_feat);
        add_feature(&mut self.acc[1], net, new_feat);
    }
}

impl Default for DualAccumulator {
    fn default() -> Self {
        Self {
            acc: [[0.0; 128]; 2],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy single-perspective Accumulator (kept for test ergonomics)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Accumulator(pub [f32; 128]);

impl Accumulator {
    pub fn new(net: &Network, board: &TicTacToe) -> Self {
        let stm_features = board.to_features();
        let mut acc = net.b0;
        for i in 0..FEATURES_COUNT {
            if stm_features[i] != 0.0 {
                for j in 0..128 {
                    acc[j] += net.w0[j][i];
                }
            }
        }
        Accumulator(acc)
    }

    pub fn add_features(&mut self, net: &Network, features: &[usize]) {
        for &f in features {
            add_feature(&mut self.0, net, f);
        }
    }

    pub fn sub_feature_single(&mut self, net: &Network, feature: usize) {
        sub_feature(&mut self.0, net, feature);
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self([0.0; 128])
    }
}
