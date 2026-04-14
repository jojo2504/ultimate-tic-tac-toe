// This script is made to auto train the network by having this loop:
// - play against it self and generate the next data.bin
// - train the data.bin into the python `train.py` and generate the next weights.bin
// - use that improved weights.bin into the next iteration of the loop

use std::process::Command;
use ultimate_tic_tac_toe::train::{self, tournament};

fn main() -> anyhow::Result<()> {
    let mut gen_count = 1;
    let mut best_net = format!("databin/gen{}_weights.bin", gen_count - 1);

    loop {
        println!("generating self-play data...");
        train::generate_iterative_databin(gen_count)?;

        println!("training new databin");
        Command::new("python")
            .arg("train.py")
            .arg(gen_count.to_string())
            .status()?;

        let challenger = format!("databin/gen{}_weights.bin", gen_count);

        println!("evaluating...");
        let elo = tournament(&best_net, &challenger, 200);
        println!("gen{gen_count} vs baseline: {elo:+.1} Elo");

        if elo > -10.0 {
            println!("promoting gen{gen_count} as new best");
            best_net = challenger;
        } else {
            println!("rejecting gen{gen_count}, keeping {best_net}");
        }

        gen_count += 1;
    }
}
