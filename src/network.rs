use bytemuck::{Pod, Zeroable, checked::try_from_bytes};
use std::fs;

use crate::{
    constants::FEATURES_COUNT,
    core::{MoveDelta, Symbol, TicTacToe},
};

fn feature_offset(symbol: Symbol, cross_off: u32, circle_off: u32) -> u32 {
    match symbol {
        Symbol::Cross => cross_off,
        Symbol::Circle => circle_off,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Network {
    w0: [[f32; FEATURES_COUNT]; 128],
    b0: [f32; 128],
    w1: [[f32; 128]; 64],
    b1: [f32; 64],
    w2: [[f32; 64]; 1],
    b2: [f32; 1],
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
        let network = Box::new(*try_from_bytes::<Network>(&bytes).expect(&format!(
            "invalid alignment or size:\nsize of bytes: {}\n",
            std::mem::size_of_val(&bytes)
        )));

        network
    }

    pub fn forward(&self, acc: &Accumulator) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.forward_avx2(acc) };
            }
        }
        self.forward_scalar(acc)
    }

    /// Scalar fallback for non-AVX2 CPUs.
    fn forward_scalar(&self, acc: &Accumulator) -> f32 {
        let mut screlu_input = [0f32; 128];
        for j in 0..128 {
            screlu_input[j] = screlu(acc.0[j]);
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

    /// AVX2 + FMA SIMD forward pass.
    /// Layer 1 (the hot loop): 64 neurons × 128 inputs = 8192 multiply-adds,
    /// done 8-wide with 4-way unrolling → 256 FMA instructions.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, acc: &Accumulator) -> f32 {
        use std::arch::x86_64::*;

        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);

        // ── Precompute screlu for all 128 accumulator values ──────────
        let mut screlu_buf = [0f32; 128];

        let stm_ptr = acc.0.as_ptr();
        let out_ptr = screlu_buf.as_mut_ptr();
        for k in (0..128).step_by(8) {
            let x = _mm256_loadu_ps(stm_ptr.add(k));
            let clamped = _mm256_min_ps(_mm256_max_ps(x, zero), one);
            _mm256_storeu_ps(out_ptr.add(k), _mm256_mul_ps(clamped, clamped));
        }

        // ── Layer 1: 64 neurons, 128 inputs, 4-way unrolled FMA ──────
        let mut h1 = [0f32; 64];
        let s_ptr = screlu_buf.as_ptr();

        for i in 0..64 {
            let w_ptr = self.w1[i].as_ptr();

            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();

            // 128 / 32 = 4 iterations
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

            // reduce 4 accumulators → scalar
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

        // ── Layer 2: 64 → 1 ─────────────────────────────────────────
        let h_ptr = h1.as_ptr();
        let w2_ptr = self.w2[0].as_ptr();

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        // 64 / 16 = 4 iterations
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

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Accumulator(pub [f32; 128]);

impl Accumulator {
    pub fn new(net: &Network, board: &TicTacToe) -> Self {
        // stm: features from current player's perspective
        let stm_features = board.to_features();

        let mut stm_acc = net.b0;

        for i in 0..FEATURES_COUNT {
            if stm_features[i] != 0.0 {
                for j in 0..128 {
                    stm_acc[j] += net.w0[j][i];
                }
            }
        }

        Accumulator(stm_acc)
    }

    pub fn add_features(&mut self, net: &Network, features: &[usize]) {
        for i in 0..128 {
            for &f in features {
                self.0[i] += net.w0[i][f];
            }
        }
    }

    pub fn sub_feature(&mut self, net: &Network, feature: usize) {
        for i in 0..128 {
            self.0[i] -= net.w0[i][feature];
        }
    }

    pub fn apply_delta(&mut self, net: &Network, delta: &MoveDelta) {
        let f = (delta.square as u32 + feature_offset(delta.turn, 0, 81)) as usize;
        self.add_features(net, &[f]);

        if let Some(new) = delta.cleared_board {
            let f1 = (new as u32 + feature_offset(delta.turn, 162, 171)) as usize;
            let f2 = 180 + new.trailing_zeros() as usize;
            self.add_features(net, &[f1, f2]);
        }

        if let Some(new) = delta.new_focus {
            let idx = 189 + new as usize;
            self.add_features(net, &[idx]);
        }
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self([0.0; 128])
    }
}
