use bytemuck::{Pod, Zeroable, checked::try_from_bytes};
use std::fs;

use crate::{
    constants::FEATURES_COUNT,
    core::{MoveDelta, Symbol, TicTacToe},
};

const N_BUCKETS: usize = 4;
const QA: i32 = 256;
const QB: i32 = 64;

// Bucket by ply (call this before forward)
pub fn get_bucket(ply: usize) -> usize {
    (ply * N_BUCKETS / 82).min(N_BUCKETS - 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Network weights
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Network {
    pub w0: [[i16; FEATURES_COUNT]; 128],
    pub b0: [i16; 128],
    pub w1: [[i16; 128]; 64],
    pub b1: [i32; 64],
    pub w2: [[i16; 64]; N_BUCKETS],
    pub b2: [i32; N_BUCKETS],
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Network {
    pub fn load(path: String) -> Box<Self> {
        let bytes = fs::read(&path).expect("failed to read weights file");
        let floats: &[f32] = bytemuck::cast_slice(&bytes);
        Box::new(Self::quantize(floats))
    }

    fn quantize(f: &[f32]) -> Self {
        let mut net = Self::zeroed();
        let mut o = 0;

        for i in 0..128 {
            for j in 0..FEATURES_COUNT {
                net.w0[i][j] = (f[o] * QA as f32).round() as i16;
                o += 1;
            }
        }
        for i in 0..128 {
            net.b0[i] = (f[o] * QA as f32).round() as i16;
            o += 1;
        }
        for i in 0..64 {
            for j in 0..128 {
                net.w1[i][j] = (f[o] * QB as f32).round() as i16;
                o += 1;
            }
        }
        for i in 0..64 {
            net.b1[i] = (f[o] * QA as f32 * QB as f32).round() as i32;
            o += 1;
        }
        for b in 0..N_BUCKETS {
            for i in 0..64 {
                net.w2[b][i] = (f[o] * QB as f32).round() as i16;
                o += 1;
            }
        }
        for b in 0..N_BUCKETS {
            net.b2[b] = (f[o] * QA as f32 * QB as f32).round() as i32;
            o += 1;
        }

        net
    }

    /// Forward pass.  `acc` is a *single-perspective* hidden layer – i.e. the
    /// caller must already have selected the STM half of a DualAccumulator.
    pub fn forward(&self, acc: &[i16; 128], bucket: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.forward_avx2(acc, bucket) };
            }
        }
        self.forward_scalar(acc, bucket)
    }

    fn forward_scalar(&self, acc: &[i16; 128], bucket: usize) -> f32 {
        // Layer 1
        let mut h1 = [0i32; 64];
        for i in 0..64 {
            let mut sum = self.b1[i]; // scale: QA * QB
            for j in 0..128 {
                let a = (acc[j] as i32).clamp(0, QA); // scale: QA
                sum += (a * a * self.w1[i][j] as i32) >> 8; // >> 8 = / QA → scale: QA * QB
            }
            h1[i] = (sum >> 6).clamp(0, QA); // >> 6 = / QB → scale: QA
        }

        // Layer 2
        let mut out = self.b2[bucket]; // scale: QA * QB
        for i in 0..64 {
            out += h1[i] * self.w2[bucket][i] as i32; // QA * QB
        }

        sigmoid(out as f32 / (QA * QB) as f32)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, acc: &[i16; 128], bucket: usize) -> f32 {
        use std::arch::x86_64::*;

        // Normalize acc to [0,1] and apply SCReLU
        let mut screlu_buf = [0f32; 128];
        for k in 0..128 {
            let a = (acc[k] as f32 / QA as f32).clamp(0.0, 1.0);
            screlu_buf[k] = a * a;
        }

        // Normalize weights to true f32 scale
        let inv_qb = 1.0 / QB as f32;
        let mut w1_f32 = [[0f32; 128]; 64];
        for i in 0..64 {
            for j in 0..128 {
                w1_f32[i][j] = self.w1[i][j] as f32 * inv_qb;
            }
        }
        let mut w2_f32 = [0f32; 64];
        for i in 0..64 {
            w2_f32[i] = self.w2[bucket][i] as f32 * inv_qb;
        }

        let s_ptr = screlu_buf.as_ptr();
        let mut h1 = [0f32; 64];

        for i in 0..64 {
            let w_ptr = w1_f32[i].as_ptr();
            // b1 stored as QA*QB scale → normalize to f32
            let bias = self.b1[i] as f32 / (QA * QB) as f32;

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
            let dot = _mm_cvtss_f32(s32) + bias; // both in [0,1] scale now ✓
            let c = dot.clamp(0.0, 1.0);
            h1[i] = c * c;
        }

        // Layer 2
        let h_ptr = h1.as_ptr();
        let w2_ptr = w2_f32.as_ptr();
        let bias2 = self.b2[bucket] as f32 / (QA * QB) as f32;

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
        sigmoid(_mm_cvtss_f32(s32) + bias2) // both [0,1] scale ✓
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn add_feature(acc: &mut [i16; 128], net: &Network, f: usize) {
    for i in 0..128 {
        acc[i] = acc[i].saturating_add(net.w0[i][f]);
    }
}

#[inline(always)]
fn sub_feature(acc: &mut [i16; 128], net: &Network, f: usize) {
    for i in 0..128 {
        acc[i] = acc[i].saturating_sub(net.w0[i][f]);
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
    pub acc: [[i16; 128]; 2],
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
    pub fn stm(&self, turn: Symbol) -> &[i16; 128] {
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
        Self { acc: [[0; 128]; 2] }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy single-perspective Accumulator (kept for test ergonomics)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Accumulator(pub [i16; 128]);

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
        Self([0; 128])
    }
}
