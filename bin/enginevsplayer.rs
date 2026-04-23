use std::{env::args, io::stdin};

use anyhow::bail;
use ultimate_tic_tac_toe::{core::TicTacToe, network::Network, search::Search};

fn input(board: &TicTacToe, mut buffer: &mut String) -> u8 {
    println!("input a valid square move 0 to 80:");

    let player_mv;
    loop {
        buffer.clear();
        stdin().read_line(&mut buffer).unwrap();
        println!("new buffer {}", buffer);
        player_mv = match buffer.trim().parse::<u8>() {
            Ok(mv) if board.validate_move(mv).is_ok() => mv,
            _ => {
                println!("Invalid input or move. Try again.");
                continue;
            }
        };
        break;
    }

    player_mv
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = args().collect();

    let net = Network::load(format!("databin/gen{}_weights.bin", args[1]));
    let mut board = TicTacToe::new();
    let mut search = Search::new();

    let mut buffer = String::new();
    println!("Who starts first ? (player || engine)");
    stdin().read_line(&mut buffer)?;

    let mut turn = match buffer.trim() {
        "player" => 0,
        "engine" => 1,
        _ => bail!("invalid option"),
    };

    println!("{}", board);
    while !board.check_win() && !board.is_full() {
        let mv = match turn {
            0 => input(&board, &mut buffer),
            1 => search.think(&board, 7, &net),
            _ => unreachable!(),
        };

        board.make(mv);
        println!("{}", board);
        turn ^= 1;
    }

    if board.check_win() {
        println!("{:?} won!", board.turn.swap());
    } else {
        println!("draw");
    }

    Ok(())
}
