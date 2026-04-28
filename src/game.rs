use crate::{
    core::TicTacToe, movegen::generate_random_legal_move, network::Network, search::Search,
    train::Sample,
};

pub fn start_self_game() {
    let net = Network::load("databin/gen0_weights.bin".to_owned());
    let mut board = TicTacToe::new();
    let mut search = Search::new();

    while !board.check_win() && !board.is_full() {
        let mv = search.think(&board, 1, &net, None);
        board.make(mv as u8);
        println!("{}", board);
    }

    println!("{:?}", board.result());
}

/// Bootstrap-only path: random policy, no search, no policy target.
/// Sample.policy is left as zeros — Phase 2 trainer must skip policy loss
/// for these samples (or weight them down).
pub fn random_game() -> Vec<Sample> {
    let mut samples = vec![];

    let mut game = TicTacToe::new();
    while !game.check_win() && !game.is_full() {
        let features = game.to_features();

        samples.push(Sample::new_no_policy(
            features,
            0.5,
            0.0, // outcome filled later
            game.ply as f32,
        ));

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
        s.outcome = if (n - 1 - i) % 2 == 0 {
            outcome
        } else {
            1.0 - outcome
        };
    }
    samples
}

/// Self-play with a trained network.
///
/// Each move now also captures the *policy distribution* produced by the
/// search, which becomes the training target for the policy head in Phase 2.
/// Dirichlet noise is injected at the root to keep self-play diverse.
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

        // Phase 1: capture (move, score, policy) instead of just (move, score)
        let (move_square, search_score, policy) =
            search.think_training_with_policy(&game, depth, net);

        pushed_samples.push(PushedSample {
            sample: Sample {
                features,
                policy,
                search_score: search_score.clamp(0.0, 1.0),
                outcome: 0.0,
                ply: game.ply as f32,
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