use crate::{core::TicTacToe, movegen::generate_random_legal_move, train::Sample};

pub fn start_random_game() {
    let mut game = TicTacToe::new();

    while !game.check_win() && !game.is_full() {
        let random_move_index = generate_random_legal_move(&game);
        game.make(random_move_index as u8);
    }

    // println!(
    //     "{:?} {:?} won: \n{}",
    //     game.check_win(),
    //     game.last_move.unwrap(),
    //     game
    // );
}

pub fn random_game(game: &mut TicTacToe) -> Vec<Sample> {
    let mut samples = vec![];

    while !game.check_win() && !game.is_full() {
        let features = game.to_features();
        samples.push(Sample {
            features,
            outcome: 0.0,
        }); // outcome filled later

        let mv = generate_random_legal_move(game);
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
