use bincode::{Encode, config};

use crate::{core::TicTacToe, game::random_game};

#[derive(Encode)]
pub struct Sample {
    pub features: [f32; 200],
    pub outcome: f32,
}

pub fn generate_databin() -> anyhow::Result<()> {
    let mut all_samples: Vec<Sample> = vec![];
    for _ in 0..10_000 {
        let mut game = TicTacToe::new();
        all_samples.extend(random_game(&mut game));
    }
    let bytes = bincode::encode_to_vec(&all_samples, config::standard()).unwrap();
    std::fs::write("databin/gen0_data.bin", bytes)?;
    Ok(())
}
