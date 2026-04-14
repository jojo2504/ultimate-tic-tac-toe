use crate::{
    core::TicTacToe, movegen::generate_random_legal_move, network::Network, search::Search,
    train::Sample,
};

pub fn start_self_game() {
    let net = Network::load("databin/gen0_weights.bin".to_owned());
    let mut board = TicTacToe::new();
    let mut search = Search::new();

    while !board.check_win() && !board.is_full() {
        let mv = search.think(&board, 1, &net);
        board.make(mv as u8);
        println!("{}", board);
    }

    println!("{:?}", board.result());

    // println!(
    //     "{:?} {:?} won: \n{}",
    //     game.check_win(),
    //     // game.last_move.unwrap(),
    //     game
    // );
}

pub fn random_game() -> Vec<Sample> {
    let mut samples = vec![];

    let mut game = TicTacToe::new();
    while !game.check_win() && !game.is_full() {
        let features = game.to_features();

        samples.push(Sample {
            features,
            outcome: 0.0,
        }); // outcome filled later

        let mv = generate_random_legal_move(&game);
        game.make(mv);
    }

    let outcome = match game.check_win() {
        true => 1.0,  // last player to move won
        false => 0.5, // draw
    };

    // alternate perspective per move
    let n = samples.len();
    for (i, s) in samples.iter_mut().enumerate() {
        // Winner moved at ply n-1; winner's positions share parity with n-1
        s.outcome = if (n - 1 - i) % 2 == 0 {
            outcome
        } else {
            1.0 - outcome
        };
    }
    samples
}

pub fn start_self_game_with_net(net: &Network) -> Vec<Sample> {
    let mut game = TicTacToe::new();
    let mut search = Search::new();

    let mut samples = vec![];
    while !game.check_win() && !game.is_full() {
        let features = game.to_features();

        samples.push(Sample {
            features,
            outcome: 0.0,
        }); // outcome filled later

        let move_square = search.think_training(&mut game, 4, &net);
        game.make(move_square);
        // println!("{}{}", game, move_square);
        // println!(
        //     "{:?}, {:09b}, {:09b}",
        //     game.turn.swap(),
        //     game.side_clear,
        //     game.all_clear
        // );
    }

    let outcome = match game.check_win() {
        true => 1.0,  // last player to move won
        false => 0.5, // draw
    };

    // println!("{:?} {}", game.turn.swap(), outcome);

    let n = samples.len();
    for (i, s) in samples.iter_mut().enumerate() {
        // Winner moved at ply n-1; winner's positions share parity with n-1
        s.outcome = if (n - 1 - i) % 2 == 0 {
            outcome
        } else {
            1.0 - outcome
        };
    }

    samples
}
