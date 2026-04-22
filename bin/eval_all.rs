use colored::Colorize;
use rayon::prelude::*;
use std::fs;
use ultimate_tic_tac_toe::{
    core::{Result, TicTacToe},
    network::Network,
    search::Search,
};

fn play_game(
    best_net: &Network,
    opp_net: &Network,
    depth: i32,
    best_is_cross: bool,
) -> (u32, u32, u32) {
    let mut game = TicTacToe::new();
    let mut best_search = Search::new();
    let mut opp_search = Search::new();

    while !game.check_win() && !game.is_full() {
        let cross_to_move = game.ply % 2 == 0;
        let best_to_move = cross_to_move == best_is_cross;

        let mv = if best_to_move {
            best_search.think(&game, depth, best_net)
        } else {
            opp_search.think(&game, depth, opp_net)
        };

        game.make(mv);
    }

    match game.result() {
        Result::Draw => (0, 1, 0),
        Result::Win => {
            let winner_was_cross = (game.ply - 1) % 2 == 0;
            if winner_was_cross == best_is_cross {
                (1, 0, 0)
            } else {
                (0, 0, 1)
            }
        }
        Result::Loss => unreachable!(),
    }
}

fn main() -> anyhow::Result<()> {
    let max_gen = 181;
    let depth = 5;

    println!("Pre-loading all available networks up to gen{}...", max_gen);
    let mut networks = std::collections::HashMap::new();
    for g in 0..=max_gen {
        let path = format!("databin/gen{}_weights.bin", g);
        if fs::metadata(&path).is_ok() {
            networks.insert(g, *Network::load(path));
        }
    }

    let mut available_gens: Vec<i32> = networks.keys().cloned().collect();
    available_gens.sort();

    let mut candidates: Vec<i32> = available_gens.iter().cloned().filter(|&g| g >= 160 && g <= 181).collect();
    candidates.sort();

    println!(
        "Starting evaluation of {} candidates (gen 160-181) against {} opponents at Depth {}...",
        candidates.len(),
        available_gens.len(),
        depth
    );

    let mut overall_results: Vec<(i32, f32, f32, f32)> = candidates
        .par_iter()
        .map(|&cand_gen| {
            let cand_net = networks.get(&cand_gen).unwrap();
            let mut total_score = 0.0;
            let mut total_games = 0.0;

            for &opp_gen in &available_gens {
                if opp_gen == cand_gen {
                    continue;
                }
                let opp_net = networks.get(&opp_gen).unwrap();

                let (w1, d1, _l1) = play_game(cand_net, opp_net, depth, true);
                let (w2, d2, _l2) = play_game(cand_net, opp_net, depth, false);

                let wins = w1 + w2;
                let draws = d1 + d2;

                total_score += wins as f32 + 0.5 * draws as f32;
                total_games += 2.0;
            }

            let winrate = (total_score / total_games) * 100.0;

            let color_str = if winrate >= 60.0 {
                format!(
                    "gen{:<3} overall: {:.1}% ({:.1} / {:.1})",
                    cand_gen, winrate, total_score, total_games
                )
                .green()
            } else if winrate < 40.0 {
                format!(
                    "gen{:<3} overall: {:.1}% ({:.1} / {:.1})",
                    cand_gen, winrate, total_score, total_games
                )
                .red()
            } else {
                format!(
                    "gen{:<3} overall: {:.1}% ({:.1} / {:.1})",
                    cand_gen, winrate, total_score, total_games
                )
                .yellow()
            };

            println!("{}", color_str);
            (cand_gen, winrate, total_score, total_games)
        })
        .collect();

    overall_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=================================================");
    println!("🏆 LEADERBOARD (Top 25 Most Robust Networks) 🏆");
    println!("=================================================");
    for (i, (g, winrate, score, games)) in overall_results.iter().take(25).enumerate() {
        println!(
            "#{:02} | gen{:<3} | {:>5.1}% Winrate | {:.1}/{:.1} Score",
            i + 1,
            g,
            winrate,
            score,
            games
        );
    }
    println!("=================================================");

    Ok(())
}
