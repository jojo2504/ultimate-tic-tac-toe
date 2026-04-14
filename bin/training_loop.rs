// This script is made to auto train the network by having this loop:
// - play against it self and generate the next data.bin
// - train the data.bin into the python `train.py` and generate the next weights.bin
// - use that improved weights.bin into the next iteration of the loop

use colored::Colorize;
use std::{fs, process::Command};
use ultimate_tic_tac_toe::train::{self, tournament};

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

        println!("evaluating...");
        let elo = tournament(&best_net, &challenger, 200);
        println!("gen{gen_count} vs baseline: {elo:+.1} Elo");

        if elo > 0.0 {
            println!(
                "{}",
                format!("promoting gen{gen_count} as new best").green()
            );
            best_net = challenger;
            best_gen = gen_count;
            upgrade_count += 1;
            if upgrade_count % 5 == 0 {
                println!("{}", "Checking if net is training well:".cyan());
                let elo = tournament(&best_net, &fixed_net, 200);
                println!("gen{gen_count} vs fixed_net: {elo:+.1} Elo");
            }
        } else {
            println!(
                "{}",
                format!("rejecting gen{gen_count}, keeping {best_net}").red()
            );
        }

        // remove old databin and weights and keep the gen 0
        if gen_count > 6 {
            fs::remove_file(format!("databin/gen{}_data.bin", gen_count - 6))?;
            fs::remove_file(format!("databin/gen{}_weights.bin", gen_count - 6))?;
        }

        gen_count += 1;
    }
}
