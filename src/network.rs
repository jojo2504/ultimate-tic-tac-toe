use bytemuck::{Pod, Zeroable, checked::try_from_bytes};
use std::fs;

use crate::constants::FEATURES_COUNT;

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

fn screlu(x: f32) -> f32 {
    x.clamp(0.0, 1.0).powi(2)
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Network {
    pub fn load(path: String) -> Box<Self> {
        let bytes = fs::read(path).expect("failed to read weights file");
        let network =
            Box::new(*try_from_bytes::<Network>(&bytes).expect("invalid alignment or size"));

        network
    }

    pub fn forward(&self, acc: AccumulatorPair) -> f32 {
        // layer 1: [128+128=256 inputs] → 64
        // note: your current w1 is [64×128] — needs to be [64×256] for dual perspective
        // for now assuming single perspective (128 inputs):
        let mut h1 = [0f32; 64];
        for i in 0..64 {
            let mut sum = self.b1[i];
            for j in 0..128 {
                sum += self.w1[i][j] * screlu(acc.stm.0[j]);
            }
            h1[i] = screlu(sum);
        }

        // layer 2: 64 → 1
        let mut out = self.b2[0];
        for i in 0..64 {
            out += self.w2[0][i] * h1[i];
        }

        sigmoid(out)
    }
}

#[derive(Clone, Copy)]
struct Accumulator([f32; 128]);

impl Accumulator {
    pub fn new(net: &Network, features: &[usize]) -> Self {
        let mut acc = net.b0; // start with bias
        for &feat in features {
            for i in 0..128 {
                acc[i] += net.w0[i][feat];
            }
        }
        Accumulator(acc)
    }

    pub fn add_feature(&mut self, net: &Network, feature: usize) {
        for i in 0..128 {
            self.0[i] += net.w0[i][feature];
        }
    }

    pub fn sub_feature() {
        unimplemented!("Not really important to implement")
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self([0.0; 128])
    }
}

#[derive(Default, Copy, Clone)]
pub struct AccumulatorPair {
    stm: Accumulator,
    nstm: Accumulator,
}

impl AccumulatorPair {
    pub fn swap_perspective(&mut self) {
        std::mem::swap(&mut self.stm, &mut self.nstm);
    }
}
