use std::{
    env::args,
    io::stdin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::bail;
use ultimate_tic_tac_toe::{
    core::TicTacToe,
    network::Network,
    search::{Search, endgame_trigger},
};

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

    let solved = Arc::new(AtomicBool::new(false));
    let stop_solver = Arc::new(AtomicBool::new(false));
    let mut solver_handle: Option<thread::JoinHandle<()>> = None;

    println!("{}", board);
    while !board.check_win() && !board.is_full() {
        // Spawn the background exact solver the first time the trigger fires.
        // From this point on it runs continuously, sharing the TT with the
        // foreground engine.
        if solver_handle.is_none() && endgame_trigger(&board) {
            println!("[endgame solver started in background]");
            let mut bg_search = search.clone();
            let bg_board = board.clone();
            let bg_stop = Arc::clone(&stop_solver);
            let bg_solved = Arc::clone(&solved);
            solver_handle = Some(thread::spawn(move || {
                bg_search.solve_background(&bg_board, bg_stop, bg_solved);
            }));
        }

        let mv = match turn {
            0 => {
                // Player's turn. The background solver, if running, fills the
                // TT freely while we wait for input. Before the trigger fires,
                // fall back to the legacy network ponder.
                if solver_handle.is_some() {
                    input(&board, &mut buffer)
                } else {
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
            }
            1 => {
                if solved.load(Ordering::Relaxed) {
                    let (mv, score) = search
                        .think_exact(&board, Duration::from_secs(1))
                        .expect("game solved but think_exact returned None");
                    println!(
                        "Played {} or ({}, {}) [solved, exact={:.3}]",
                        mv,
                        mv / 9 + 1,
                        mv % 9 + 1,
                        score
                    );
                    mv
                } else {
                    println!("Engine is thinking...");
                    let result = search.iterative_deepening_think(&board, &net, Duration::from_secs(3));
                    let mv = result.0.unwrap();
                    println!(
                        "Played {} or ({}, {}) with depth: {}",
                        mv,
                        mv / 9 + 1,
                        mv % 9 + 1,
                        result.1
                    );
                    mv
                }
            }
            _ => unreachable!(),
        };

        board.make(mv);
        println!("{}", board);
        turn ^= 1;
    }

    // Cleanly terminate the background solver if it is still running.
    stop_solver.store(true, Ordering::Relaxed);
    if let Some(h) = solver_handle {
        let _ = h.join();
    }

    if board.check_win() {
        println!("{:?} won!", board.turn.swap());
    } else {
        println!("draw");
    }

    Ok(())
}
