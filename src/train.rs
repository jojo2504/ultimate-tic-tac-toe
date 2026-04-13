use rayon::prelude::*;
use std::io::{BufWriter, Write};

use bincode::Encode;

use crate::{
    constants::FEATURES_COUNT,
    core::TicTacToe,
    game::{random_game, start_self_game_with_net},
    movegen::generate_moves,
    network::Network,
    search::Search,
};

#[derive(Encode)]
pub struct Sample {
    pub features: [f32; FEATURES_COUNT * 2],
    pub outcome: f32,
}

pub fn generate_first_databin(gen_count: i32) -> anyhow::Result<()> {
    let mut all_samples: Vec<Sample> = vec![];
    let games_samples: Vec<Vec<Sample>> = (0..1000)
        .into_par_iter()
        .map(|i| {
            let samples = random_game();
            println!("game {} completed.", i);
            samples
        })
        .collect();

    for samples in games_samples {
        all_samples.extend(samples);
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

pub fn generate_iterative_databin(gen_count: i32) -> anyhow::Result<()> {
    let mut all_samples: Vec<Sample> = vec![];
    let net = Network::load(format!("databin/gen{}_weights.bin", gen_count - 1));
    // let mut search = Search::new();
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let games_samples: Vec<Vec<Sample>> = (0..1000)
        .into_par_iter()
        .map(|_i| {
            let samples = start_self_game_with_net(&net);
            let count = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if count % 100 == 0 {
                println!("{} games completed.", count);
            }
            samples
        })
        .collect();

    for samples in games_samples {
        all_samples.extend(samples);
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
