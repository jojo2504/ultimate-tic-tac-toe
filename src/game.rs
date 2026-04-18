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
            search_score: 0.5,
            outcome: 0.0,
            ply: game.ply,
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

pub fn start_self_game_with_net(net: &Network, depth: i32) -> Vec<Sample> {
    let mut game = TicTacToe::new();
    let mut search = Search::new();

    struct PushedSample {
        sample: Sample,
        ply: usize,
    }
    let mut pushed_samples = vec![];

    while !game.check_win() && !game.is_full() {
        let features = game.to_features();
        let ply = game.ply;

        let (move_square, search_score) = search.think_training_scored(&game, depth, &net);

        pushed_samples.push(PushedSample {
            sample: Sample {
                features,
                search_score: search_score.clamp(0.0, 1.0),
                outcome: 0.0,
                ply: game.ply,
            },
            ply,
        });

        let delta = game.make(move_square);

        if delta.cleared_board.is_some() || ply < 6 {
            pushed_samples.pop();
        }
    }

    let outcome = match game.check_win() {
        true => 1.0,
        false => 0.5,
    };

    let final_ply = game.ply;
    let mut final_samples = vec![];

    for mut ps in pushed_samples {
        ps.sample.outcome = if (final_ply - 1 - ps.ply) % 2 == 0 {
            outcome
        } else {
            1.0 - outcome
        };
        final_samples.push(ps.sample);
    }

    final_samples
}
