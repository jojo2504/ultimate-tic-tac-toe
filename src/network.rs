use bytemuck::{Pod, Zeroable, checked::try_from_bytes};
use std::fs;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Network {
    w0: [[f32; 200]; 128],
    b0: [f32; 128],
    w1: [[f32; 128]; 64],
    b1: [f32; 64],
    w2: [[f32; 64]; 1],
    b2: [f32; 1],
}

impl Network {
    pub fn load(path: String) -> Box<Self> {
        let bytes = fs::read(path).expect("failed to read weights file");
        let network =
            Box::new(*try_from_bytes::<Network>(&bytes).expect("invalid alignment or size"));

        network
    }

    pub fn relu(weight: f32) -> f32 {
        weight.max(0.0)
    }

    pub fn sigmoid(sum: f32) -> f32 {
        1.0 / (1.0 + (-sum).exp())
    }

    pub fn predict() {
        todo!()
    }
}
