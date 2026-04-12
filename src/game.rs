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
        let mut features = [0.0; crate::constants::FEATURES_COUNT * 2];
        let stm_features = game.to_features();

        let mut flipped = game.clone();
        flipped.turn = flipped.turn.swap();
        let nstm_features = flipped.to_features();

        features[..crate::constants::FEATURES_COUNT].copy_from_slice(&stm_features);
        features[crate::constants::FEATURES_COUNT..].copy_from_slice(&nstm_features);

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
        s.outcome = if (n - i) % 2 == 0 {
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
        let mut features = [0.0; crate::constants::FEATURES_COUNT * 2];
        let stm_features = game.to_features();

        let mut flipped = game.clone();
        flipped.turn = flipped.turn.swap();
        let nstm_features = flipped.to_features();

        features[..crate::constants::FEATURES_COUNT].copy_from_slice(&stm_features);
        features[crate::constants::FEATURES_COUNT..].copy_from_slice(&nstm_features);

        samples.push(Sample {
            features,
            outcome: 0.0,
        }); // outcome filled later

        let move_square = search.think_training(&mut game, 4, &net);
        game.make(move_square);
        println!("{}", game);
    }

    let outcome = match game.check_win() {
        true => 1.0,  // last player to move won
        false => 0.5, // draw
    };

    let n = samples.len();
    for (i, s) in samples.iter_mut().enumerate() {
        s.outcome = if (n - i) % 2 == 0 {
            outcome
        } else {
            1.0 - outcome
        };
    }

    samples
}
