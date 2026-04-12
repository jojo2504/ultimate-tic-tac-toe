use ultimate_tic_tac_toe::{core::TicTacToe, network::Network, search::Search};

fn main() {
    // start_self_game();
    let net = Network::load("databin/gen0_weights.bin".to_owned());
    let mut board = TicTacToe::new();
    let mut search = Search::new();
    let mv = search.think(&board, 1, &net);
    board.make(mv);
    println!("{}", board);
    // let mv = search.think(&board, 1, &net);
    // board.make(mv);
    // println!("{}", board);
    // let mv = search.think(&board, 1, &net);
    // board.make(mv);
    // println!("{}", board);

    // let random_move_index = search.think(&mut game, 4, &net);
    // println!("{}", game);
    //
    //     let mut board = TicTacToe::new();
    //     board.make(1 as u8);
    //     println!("{}", board);
    //     board.make(3 as u8);
    //     println!("{}", board);
}
