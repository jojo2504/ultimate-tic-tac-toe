use rayon::prelude::*;
use std::io::{BufWriter, Write};

use bincode::Encode;

use crate::{
    constants::FEATURES_COUNT,
    core::{Result, TicTacToe},
    game::{random_game, start_self_game_with_net},
    network::{DualAccumulator, Network},
    search::Search,
};

#[derive(Encode)]
pub struct Sample {
    pub features: [f32; FEATURES_COUNT],
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

    flush_samples(&all_samples, gen_count)
}

pub fn generate_iterative_databin(gen_count: i32, best_gen: i32, depth: i32) -> anyhow::Result<()> {
    let mut all_samples: Vec<Sample> = vec![];
    let net = Network::load(format!("databin/gen{}_weights.bin", best_gen));
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let games_samples: Vec<Vec<Sample>> = (0..1000)
        .into_par_iter()
        .map(|_i| {
            let samples = start_self_game_with_net(&net, depth);
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

    flush_samples(&all_samples, gen_count)
}

fn flush_samples(samples: &[Sample], gen_count: i32) -> anyhow::Result<()> {
    let file = std::fs::File::create(format!("databin/gen{}_data.bin", gen_count))?;
    let mut writer = BufWriter::new(file);
    for s in samples {
        for f in &s.features {
            writer.write_all(&f.to_le_bytes())?;
        }
        writer.write_all(&s.outcome.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn tournament(
    base_net_path: &str,
    challenger_net_path: &str,
    num_games: u32,
    depth: i32,
) -> f32 {
    let base_net = Network::load(base_net_path.to_owned());
    let challenger_net = Network::load(challenger_net_path.to_owned());

    let results: Vec<(u32, u32, u32)> = (0..num_games)
        .into_par_iter()
        .map(|game_index| {
            let mut game = TicTacToe::new();
            let mut challenger_search = Search::new();
            let mut base_search = Search::new();

            let challenger_is_cross = game_index % 2 == 0;

            // Initialise accumulator stacks for both engines from the root position
            challenger_search.acc[0] = DualAccumulator::new(&challenger_net, &game);
            base_search.acc[0] = DualAccumulator::new(&base_net, &game);

            while !game.check_win() && !game.is_full() {
                let cross_to_move = game.ply % 2 == 0;
                let challenger_to_move = cross_to_move == challenger_is_cross;

                if challenger_to_move {
                    let mv = challenger_search.think_training(&game, depth, &challenger_net);
                    let old_ply = game.ply;
                    let delta = game.make(mv);
                    challenger_search.acc[game.ply] = challenger_search.acc[old_ply];
                    challenger_search.acc[game.ply].apply_delta(&challenger_net, &delta);
                } else {
                    let mv = base_search.think_training(&game, depth, &base_net);
                    let old_ply = game.ply;
                    let delta = game.make(mv);
                    base_search.acc[game.ply] = base_search.acc[old_ply];
                    base_search.acc[game.ply].apply_delta(&base_net, &delta);
                }
            }

            match game.result() {
                Result::Draw => (0, 1, 0),
                Result::Win => {
                    let winner_was_cross = (game.ply - 1) % 2 == 0;
                    if winner_was_cross == challenger_is_cross {
                        (1, 0, 0)
                    } else {
                        (0, 0, 1)
                    }
                }
                Result::Loss => unreachable!(),
            }
        })
        .collect();

    let (mut wins, mut draws, mut losses) = (0u32, 0u32, 0u32);
    for (w, d, l) in results {
        wins += w;
        draws += d;
        losses += l;
    }

    let elo = elo_diff(wins, draws, losses);
    println!("\nfinal: {wins}W {draws}D {losses}L → {elo:+.1} Elo");
    elo
}

fn elo_diff(wins: u32, draws: u32, losses: u32) -> f32 {
    let total = (wins + draws + losses) as f32;
    if total == 0.0 {
        return 0.0;
    }
    let score = (wins as f32 + 0.5 * draws as f32) / total;
    if score <= 0.0 {
        return -1000.0;
    }
    if score >= 1.0 {
        return 1000.0;
    }
    -400.0 * (1.0 / score - 1.0).log10()
}
