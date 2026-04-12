// This script is made to auto train the network by having this loop:
// - play against it self and generate the next data.bin
// - train the data.bin into the python `train.py` and generate the next weights.bin
// - use that improved weights.bin into the next iteration of the loop

use std::process::Command;

use ultimate_tic_tac_toe::{
    core::{Result, TicTacToe},
    network::Network,
    search::Search,
    train,
};

pub fn tournament(base_net_path: &str, challenger_net_path: &str, num_games: u32) -> f32 {
    let base_net = Network::load(base_net_path.to_owned());
    let challenger_net = Network::load(challenger_net_path.to_owned());
    let mut search = Search::new();

    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;

    for game_index in 0..num_games {
        let mut game = TicTacToe::new();

        // alternate colors — challenger is Cross (first) on even games
        let challenger_is_cross = game_index % 2 == 0;

        while !game.is_game_over() {
            // Cross always moves on even plies, Circle on odd plies
            let cross_to_move = game.ply % 2 == 0;
            let challenger_to_move = cross_to_move == challenger_is_cross;

            let mv = if challenger_to_move {
                search.think(&game, 5, &challenger_net)
            } else {
                search.think(&game, 5, &base_net)
            };

            game.make(mv);
        }

        match game.result() {
            Result::Draw => {
                draws += 1;
                println!("game {}/{}: Draw", game_index + 1, num_games);
            }
            Result::Win => {
                // winner = player who just moved = was on ply (game.ply - 1)
                let winner_was_cross = (game.ply - 1) % 2 == 0;
                if winner_was_cross == challenger_is_cross {
                    wins += 1;
                    println!("game {}/{}: Challenger wins", game_index + 1, num_games);
                } else {
                    losses += 1;
                    println!("game {}/{}: Baseline wins", game_index + 1, num_games);
                }
            }
            Result::Loss => unreachable!(),
        }
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

fn main() -> anyhow::Result<()> {
    let mut gen_count = 1;
    let mut best_net = format!("gen{}_weights.bin", gen_count - 1);

    loop {
        println!("generating self-play data...");
        train::generate_iterative_databin(gen_count)?;

        println!("training new databin");
        Command::new("python")
            .arg("train.py")
            .arg(gen_count.to_string())
            .status()?;

        let challenger = format!("databin/gen{gen_count}_weights.bin");

        println!("evaluating...");
        let elo = tournament(&challenger, &best_net, 200);
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
