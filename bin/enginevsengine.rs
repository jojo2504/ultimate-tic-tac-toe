use std::env::args;

use ultimate_tic_tac_toe::{core::TicTacToe, network::Network, search::Search};

fn main() {
    let args: Vec<String> = args().collect();

    let engine1_path = format!("databin/gen{}_weights.bin", args[1]);
    let engine2_path = format!("databin/gen{}_weights.bin", args[2]);

    println!("Loading Engine 1 (Cross): {}", engine1_path);
    let net1 = Network::load(engine1_path.to_string());

    println!("Loading Engine 2 (Circle): {}", engine2_path);
    let net2 = Network::load(engine2_path.to_string());

    let mut board = TicTacToe::new();
    let mut search1 = Search::new();
    let mut search2 = Search::new();

    let depth = args[3];

    println!("Starting game at depth {}...", depth);
    println!("{}", board);

    while !board.check_win() && !board.is_full() {
        let cross_to_move = board.ply % 2 == 0;

        let mv = if cross_to_move {
            println!("Engine 1 (Cross) is thinking...");
            search1.think(&board, depth, &net1)
        } else {
            println!("Engine 2 (Circle) is thinking...");
            search2.think(&board, depth, &net2)
        };

        board.make(mv);
        println!("Move played: {}", mv);
        println!("{}", board);
    }

    println!("Game over!");
    if board.check_win() {
        if (board.ply - 1) % 2 == 0 {
            println!("Result: Engine 1 (Cross) wins!");
        } else {
            println!("Result: Engine 2 (Circle) wins!");
        }
    } else {
        println!("Result: Draw!");
    }
}
