use std::{
    env::args,
    io::stdin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

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
    println!("set engine to depth: {}", args[2]);
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
            0 => {
                let stop_pondering = Arc::new(AtomicBool::new(false));
                let ponder_signal = Arc::clone(&stop_pondering);
                let mut ponder_search = search.clone();
                let ponder_board = board.clone();
                let ponder_net = net.clone();

                let ponder_handle = thread::spawn(move || {
                    ponder_search.iterative_deepening(&ponder_board, &ponder_net, stop_pondering);
                });

                let player_move = input(&board, &mut buffer);
                ponder_signal.store(true, Ordering::Relaxed);
                let _ = ponder_handle.join();

                player_move
            }
            1 => {
                println!("Engine is thinking...");
                search.think(&board, args[2].parse::<i32>().unwrap(), &net, None)
            }
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
