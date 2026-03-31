use crate::{core::TicTacToe, game::start_random_game};

pub mod core;
pub mod game;
pub mod movegen;
pub mod search;

fn main() {
    start_random_game();

    // let mut game = TicTacToe::new();
    // game.make(30);
    // println!("{}", game);
    // println!("{:?}", game.state.current_focus);
}
