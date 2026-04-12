use std::io::{BufWriter, Write};

use bincode::Encode;

use crate::{constants::FEATURES_COUNT, game::random_game};

#[derive(Encode)]
pub struct Sample {
    pub features: [f32; FEATURES_COUNT * 2],
    pub outcome: f32,
}

pub fn generate_databin(gen_count: i32) -> anyhow::Result<()> {
    let mut all_samples: Vec<Sample> = vec![];
    for _ in 0..10_000 {
        all_samples.extend(random_game());
    }

    let file = std::fs::File::create(format!("databin/gen{}_data.bin", gen_count))?;
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
