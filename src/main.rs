use crate::{core::TicTacToe, game::start_random_game};
use std::time::{self, Duration, Instant};

pub mod core;
pub mod game;
pub mod movegen;
pub mod search;
pub mod train;

fn main() -> anyhow::Result<()> {
    train::generate_databin()?;
    Ok(())
}
