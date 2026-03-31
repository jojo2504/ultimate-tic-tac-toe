use crate::{core::TicTacToe, movegen::generate_moves};
use rand::random_range;

pub fn start_random_game() {
    let mut game = TicTacToe::new();

    while !game.check_win() {
        let moves: u128 = generate_moves(&game);
        let count = moves.count_ones();
        if count == 0 {
            println!("draw");
            return;
        }
        let pick = random_range(0..count);

        // skip `pick` set bits
        let mut remaining = moves;
        for _ in 0..pick {
            remaining &= remaining - 1; // clear lowest set bit
        }
        let random_move = 1u128 << remaining.trailing_zeros();
        let random_move_index = random_move.trailing_zeros();
        game.make(random_move_index as u8);
        println!("{}", game);
    }

    println!(
        "{:?} {:?} won: \n{}",
        game.state.player_turn.swap(),
        game.state.last_move.unwrap(),
        game
    );
}
