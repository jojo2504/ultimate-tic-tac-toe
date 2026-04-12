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

fn tournament(base_net: &str, challenger_net: &str) {
    let base_net = Network::load(base_net.to_owned());
    let challenger_net = Network::load(challenger_net.to_owned());
    let mut game = TicTacToe::new();
    let mut search = Search::new();

    let ply = 0;
    while !game.check_win() && !game.is_full() {
        let move_square = if ply & 1 == 0 {
            search.think(&mut game, 5, &base_net)
        } else {
            search.think(&mut game, 5, &challenger_net)
        };
        game.make(move_square);
        println!("{}", game);
    }

    match game.result() {
        Result::Win => todo!(),
        Result::Loss => todo!(),
        Result::Draw => todo!(),
    }
}

fn main() -> anyhow::Result<()> {
    let best_net = "gen0_weights.bin";
    let mut gen_count = 1;

    loop {
        train::generate_databin(gen_count)?;
        let output = Command::new("python")
            .arg("train.py")
            .arg(gen_count.to_string());

        // tournament to determine which net is the best
        tournament(best_net, &format!("gen{}_weights.bin", gen_count));

        gen_count += 1;
    }
}
