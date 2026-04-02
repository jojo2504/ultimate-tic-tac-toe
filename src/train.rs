use std::io::{BufWriter, Write};

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

    let file = std::fs::File::create("databin/gen0_data.bin")?;
    let mut writer = BufWriter::new(file);

    for s in &all_samples {
        for f in &s.features {
            writer.write_all(&f.to_le_bytes())?;
        }
        writer.write_all(&s.outcome.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
