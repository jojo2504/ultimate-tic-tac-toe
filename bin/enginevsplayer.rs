use std::io::stdin;

use ultimate_tic_tac_toe::{core::TicTacToe, network::Network, search::Search};

fn input() -> String {
    let mut result = String::new();
    stdin().read_line(&mut result).unwrap();
    result
}

fn main() {
    let net = Network::load("databin/gen11_weights.bin".to_owned());
    let mut board = TicTacToe::new();
    let mut search = Search::new();

    println!("{}", board);
    while !board.check_win() && !board.is_full() {
        println!("input a valid square move 0 to 80:");
        let player_mv = match input().trim().parse::<u8>() {
            Ok(mv) if board.validate_move(mv).is_ok() => mv,
            _ => {
                println!("Invalid input or move. Try again.");
                continue;
            }
        };
        board.make(player_mv);
        println!("{}", board);

        let engine_mv = search.think(&board, 7, &net);
        board.make(engine_mv);
        println!("{}", board);
    }

    if board.check_win() {
        println!("{:?} won!", board.turn.swap());
    } else {
        println!("draw");
    }
}
