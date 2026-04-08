use std::io::{self, BufRead, Write};
use ultimate_tic_tac_toe::{core::TicTacToe, movegen::generate_random_legal_move};

fn main() {
    let mut game = TicTacToe::new();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // handshake
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        match line.trim() {
            "utttci" => {
                writeln!(out, "id name SelfPlay").unwrap();
                writeln!(out, "id author me").unwrap();
                writeln!(out, "utttciok").unwrap();
            }
            "isready" => {
                writeln!(out, "readyok").unwrap();
            }
            "newgame" => {
                game = TicTacToe::new();
            }
            "newgame black" => {
                game = TicTacToe::new();
            }
            "go" => {
                // X moves first — pick any legal square
                let cell = first_legal(&game);
                game.make(cell);
                writeln!(out, "{cell}").unwrap();
            }
            "quit" => break,
            other => {
                // opponent square
                if let Ok(cell) = other.parse::<u8>() {
                    game.make(cell);
                    let reply = generate_random_legal_move(&game);
                    game.make(reply);
                    writeln!(out, "{reply}").unwrap();
                }
            }
        }
        out.flush().unwrap();
    }
}

fn first_legal(game: &TicTacToe) -> u8 {
    for cell in 0u8..81 {
        if game.validate_move(cell).is_ok() {
            return cell;
        }
    }
    panic!("no legal moves");
}
