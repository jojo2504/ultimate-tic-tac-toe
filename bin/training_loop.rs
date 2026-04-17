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

/// Remove the oldest generation files (data + weights) to keep disk usage under the cap.
/// Always keeps gen0 files. Returns how many generations were removed.
fn cleanup_old_generations(max_bytes: u64) -> usize {
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

    // Remove oldest first until we're under the cap
    for gen_num in gens {
        if databin_size_bytes() <= max_bytes {
            break;
        }
        let data_path = format!("databin/gen{}_data.bin", gen_num);
        let weights_path = format!("databin/gen{}_weights.bin", gen_num);
        let _ = fs::remove_file(&data_path);
        let _ = fs::remove_file(&weights_path);
        println!("{}", format!("cleaned up gen{gen_num} (disk cap)").yellow());
        removed += 1;
    }
    removed
}

const MAX_DATABIN_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GB

fn main() -> anyhow::Result<()> {
    let mut gen_count = 1;
    let mut best_gen = 0;
    let mut best_net = format!("databin/gen{}_weights.bin", gen_count - 1);
    let fixed_net = format!("databin/gen0_weights.bin"); // this fixed net is to measure how well and confirming our network is training
    let mut upgrade_count = 0;
    loop {
        println!("generating self-play data...");
        train::generate_iterative_databin(gen_count, best_gen)?;

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
            let elo = tournament(&past_net, &challenger, 400);
            println!("gen{gen_count} vs gen{past_gen}: {elo:+.1} Elo");
            total_elo += elo;
        }

        let avg_elo = total_elo / pool.len() as f32;
        println!("gen{gen_count} vs pool average: {avg_elo:+.1} Elo");

        if avg_elo > 0.0 {
            println!(
                "{}",
                format!("promoting gen{gen_count} as new best").green()
            );
            best_net = challenger;
            best_gen = gen_count;
            upgrade_count += 1;
            println!("{}", "Checking if net is training well:".cyan());
            let elo = tournament(&best_net, &fixed_net, 500);
            println!("gen{gen_count} vs fixed_net: {elo:+.1} Elo");
        } else {
            println!(
                "{}",
                format!("rejecting gen{gen_count}, keeping {best_net}").red()
            );
        }

        // Smart disk cleanup: remove oldest generations when over 10 GB cap
        let removed = cleanup_old_generations(MAX_DATABIN_BYTES);
        if removed > 0 {
            let size_mb = databin_size_bytes() as f64 / (1024.0 * 1024.0);
            println!(
                "{}",
                format!("removed {removed} old gen(s), databin now {size_mb:.0} MB").yellow()
            );
        }

        gen_count += 1;
    }
}
