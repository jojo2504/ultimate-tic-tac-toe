// This script is made to auto train the network by having this loop:
// - play against it self and generate the next data.bin
// - train the data.bin into the python `train.py` and generate the next weights.bin
// - use that improved weights.bin into the next iteration of the loop

use colored::Colorize;
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
fn cleanup_old_generations(current_gen: i32) -> usize {
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

    // Remove data files older than current_gen - 20
    for gen_num in gens {
        if gen_num >= current_gen - 20 {
            break;
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
    let mut gen_count = 177;
    let mut best_gen = 175;
    let mut best_net = format!("databin/gen{}_weights.bin", gen_count - 1);
    let fixed_net = format!("databin/gen0_weights.bin"); // this fixed net is to measure how well and confirming our network is training

    let mut depth = 3;
    let mut plateau_count = 0;
    let mut global_elo = 1200.0;

    loop {
        println!("generating self-play data... (depth {depth})");
        train::generate_iterative_databin(gen_count, best_gen, depth)?;

        println!("training new databin");
        Command::new("python")
            .arg("train.py")
            .arg(gen_count.to_string())
            .arg("--base-weights")
            .arg(&best_net)
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
            let elo = tournament(&past_net, &challenger, 400, depth);
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

        if plateau_count >= 4 {
            depth += 1;
            plateau_count = 0;
            println!(
                "{}",
                format!(
                    "\n>>> REJECTIONS PLATEAUED. AUTOMATICALLY BUMPING TRAINING DEPTH TO {} <<<\n",
                    depth
                )
                .magenta()
                .bold()
            );
        }

        if promoted {
            global_elo += elo_vs_best;
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
            println!("{}", "Checking if net is training well:".cyan());
            let elo = tournament(&fixed_net, &best_net, 500, depth);
            println!("gen{gen_count} vs fixed_net: {elo:+.1} Elo");
        } else {
            println!(
                "{}",
                format!("rejecting gen{gen_count}, keeping {best_net}").red()
            );
        }

        // Smart disk cleanup: remove data files outside the 20 gen window
        let removed = cleanup_old_generations(gen_count);
        if removed > 0 {
            let size_mb = databin_size_bytes() as f64 / (1024.0 * 1024.0);
            println!(
                "{}",
                format!("removed {removed} old data file(s), databin now {size_mb:.0} MB").yellow()
            );
        }

        gen_count += 1;
    }
}
