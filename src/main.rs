use crate::{core::TicTacToe, game::start_random_game};
use std::time::{self, Duration, Instant};

pub mod core;
pub mod game;
pub mod movegen;
pub mod search;

fn main() {
    let now = Instant::now();
    let limit = Duration::from_secs(1);
    let mut i = 0;
    while Instant::now() - now < limit {
        start_random_game();
        i += 1;
    }

    println!("{}", i);

    // let mut game = TicTacToe::new();
    // game.make(30);
    // println!("{}", game);
    // println!("{:?}", game.state.current_focus);
}
