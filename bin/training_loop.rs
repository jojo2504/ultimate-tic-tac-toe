// This script is made to auto train the network by having this loop:
// - play against it self and generate the next data.bin
// - train the data.bin into the python `train.py` and generate the next weights.bin
// - use that improved weights.bin into the next iteration of the loop

use colored::Colorize;
use serde_json::Value;
use std::{fs, process::Command};
use ultimate_tic_tac_toe::train::{self, tournament};

/// Get the total size of the databin/ directory in bytes.
fn databin_size_bytes() -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir("databin") {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

/// Remove data files outside the 20-generation window to save disk space.
/// Always keeps gen0 files and ALL weights files. Returns how many were removed.
fn cleanup_old_generations(current_gen: i32, window_length: i32) -> usize {
    let mut removed = 0;
    // Collect existing generation numbers (excluding gen0)
    let mut gens: Vec<i32> = vec![];
    if let Ok(entries) = fs::read_dir("databin") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("gen") && name.ends_with("_data.bin") {
                if let Ok(num) = name
                    .trim_start_matches("gen")
                    .trim_end_matches("_data.bin")
                    .parse::<i32>()
                {
                    if num > 0 {
                        gens.push(num);
                    }
                }
            }
        }
    }
    gens.sort();

    // Remove data files older than current_gen - window_length, keeping every 10-milestone
    for gen_num in gens {
        if gen_num >= current_gen - window_length || gen_num % 10 == 0 {
            continue;
        }
        let data_path = format!("databin/gen{}_data.bin", gen_num);
        if fs::metadata(&data_path).is_ok() {
            let _ = fs::remove_file(&data_path);
            println!(
                "{}",
                format!("cleaned up gen{gen_num} data (out of window)").yellow()
            );
            removed += 1;
        }
    }
    removed
}

fn main() -> anyhow::Result<()> {
    let mut gen_count = 1;
    let mut best_gen = 0;
    let mut best_net = format!("databin/gen{}_weights.bin", gen_count - 1);

    let depth = 3;
    let mut plateau_count = 0;
    let mut global_elo = fs::read_to_string("databin/global_elo.txt")
        .unwrap_or_else(|_| "1200.0".to_string())
        .trim()
        .parse::<f32>()
        .unwrap_or(1200.0);

    loop {
        let mut games_per_generation = 3000;
        let mut window_length = 35;
        let mut plateau_threshold = 4;

        if let Ok(config_str) = fs::read_to_string("config/config.json") {
            if let Ok(config) = serde_json::from_str::<Value>(&config_str) {
                if let Some(p) = config.get("plateau").and_then(|v| v.as_i64()) {
                    plateau_threshold = p as i32;
                }
                if let Some(depth_config) =
                    config.get("depth").and_then(|d| d.get(depth.to_string()))
                {
                    if let Some(g) = depth_config
                        .get("games_per_generation")
                        .and_then(|v| v.as_i64())
                    {
                        games_per_generation = g as i32;
                    }
                    if let Some(w) = depth_config.get("window_length").and_then(|v| v.as_i64()) {
                        window_length = w as i32;
                    }
                }
            }
        }

        println!("generating self-play data... (depth {depth}, games {games_per_generation})");
        train::generate_iterative_databin(gen_count, best_gen, depth, games_per_generation)?;

        println!("training new databin");
        Command::new("python")
            .arg("train.py")
            .arg(gen_count.to_string())
            .arg("--base-weights")
            .arg(&best_net)
            .arg("--depth")
            .arg(depth.to_string())
            .status()?;

        let challenger = format!("databin/gen{}_weights.bin", gen_count);

        println!("evaluating against pool...");
        let mut total_elo = 0.0;
        let mut elo_vs_best = 0.0;

        let mut pool = vec![best_gen];
        if best_gen >= 1 {
            pool.push(best_gen.saturating_sub(1));
        }
        if best_gen >= 3 {
            pool.push(best_gen / 2);
        }
        if best_gen >= 5 {
            pool.push(0);
        }
        pool.sort_unstable();
        pool.dedup();
        pool.retain(|&g| fs::metadata(format!("databin/gen{}_weights.bin", g)).is_ok());

        for &past_gen in &pool {
            let past_net = format!("databin/gen{}_weights.bin", past_gen);
            let elo = tournament(&past_net, &challenger, 200, depth);
            println!("gen{gen_count} vs gen{past_gen}: {elo:+.1} Elo");
            total_elo += elo;
            if past_gen == best_gen {
                elo_vs_best = elo;
            }
        }

        let avg_elo = total_elo / pool.len() as f32;
        println!("gen{gen_count} vs pool average: {avg_elo:+.1} Elo");

        let promoted = avg_elo > 0.0 && elo_vs_best > 0.0;

        if !promoted {
            plateau_count += 1;
        } else {
            plateau_count = 0;
        }

        if plateau_count >= plateau_threshold {
            plateau_count = 0;
            println!(
                "{}",
                format!(
                    "\n>>> REJECTIONS PLATEAUING. BE CAREFUL <<< {} \n",
                    plateau_count
                )
                .magenta()
                .bold()
            );
            break;
        }

        if promoted {
            global_elo += elo_vs_best;
            let _ = fs::write("databin/global_elo.txt", global_elo.to_string());
            println!(
                "{}",
                format!(
                    "promoting gen{gen_count} as new best (Global Elo: {:.1})",
                    global_elo
                )
                .green()
            );
            best_net = challenger;
            best_gen = gen_count;
        } else {
            println!(
                "{}",
                format!("rejecting gen{gen_count}, keeping {best_net}").red()
            );
        }

        // Smart disk cleanup: remove data files outside the window
        let removed = cleanup_old_generations(gen_count, window_length);
        if removed > 0 {
            let size_mb = databin_size_bytes() as f64 / (1024.0 * 1024.0);
            println!(
                "{}",
                format!("removed {removed} old data file(s), databin now {size_mb:.0} MB").yellow()
            );
        }

        gen_count += 1;
    }

    Ok(())
}
