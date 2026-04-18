use anyhow::{Context, Result, bail};
use std::{
    env,
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};
use ultimate_tic_tac_toe::core::TicTacToe;

// ── Protocol ─────────────────────────────────────────────────────────────────
//
// HOST → ENGINE
//   utttci              handshake
//   isready             ping
//   newgame             reset, you are X (first to move)
//   newgame black       reset, you are O (second to move)
//   <0-80>              opponent just played this square; reply with your square
//   go                  you move first (only on turn 1 for X); reply with your square
//   quit
//
// ENGINE → HOST
//   utttciok
//   id name <name>
//   id author <author>
//   readyok
//   <0-80>              your chosen square
// ─────────────────────────────────────────────────────────────────────────────

struct Engine {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pub name: String,
}

impl Engine {
    fn spawn(path: &str) -> Result<Self> {
        let mut process = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn engine: {path}"))?;

        let stdin = process.stdin.take().context("no stdin")?;
        let stdout = BufReader::new(process.stdout.take().context("no stdout")?);

        let mut engine = Self {
            process,
            stdin,
            stdout,
            name: path.to_string(),
        };

        engine.handshake()?;
        Ok(engine)
    }

    // ── low-level I/O ───────────────────────────────────────────────────────

    fn send(&mut self, cmd: &str) -> Result<()> {
        writeln!(self.stdin, "{cmd}")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_square(&mut self) -> Result<u8> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line)?;
            if n == 0 {
                bail!("engine '{}' closed stdout unexpectedly", self.name);
            }
            let trimmed = line.trim();
            // skip info lines
            if trimmed.starts_with("info") {
                eprintln!("[{}] {}", self.name, trimmed);
                continue;
            }
            let cell: u8 = trimmed
                .parse()
                .with_context(|| format!("engine '{}' sent non-integer: '{trimmed}'", self.name))?;
            if cell > 80 {
                bail!("engine '{}' replied {cell}, out of range 0-80", self.name);
            }
            return Ok(cell);
        }
    }

    // ── protocol commands ───────────────────────────────────────────────────

    fn handshake(&mut self) -> Result<()> {
        self.send("utttci")?;
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line)?;
            let trimmed = line.trim().to_string();
            if trimmed.starts_with("id name ") {
                self.name = trimmed["id name ".len()..].to_string();
            }
            if trimmed == "utttciok" {
                break;
            }
        }
        self.send("isready")?;
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line)?;
            if line.trim() == "readyok" {
                break;
            }
        }
        Ok(())
    }

    /// Tell the engine it's a new game and whether it plays first or second.
    fn new_game(&mut self, plays_first: bool) -> Result<()> {
        if plays_first {
            self.send("newgame")
        } else {
            self.send("newgame black")
        }
    }

    /// Notify the engine of the opponent's move and get its reply.
    fn send_opponent_move(&mut self, square: u8) -> Result<u8> {
        self.send(&square.to_string())?;
        self.read_square()
    }

    /// Ask the engine to move first (only used for X on turn 1).
    fn go_first(&mut self) -> Result<u8> {
        self.send("go")?;
        self.read_square()
    }

    fn quit(&mut self) -> Result<()> {
        let _ = self.send("quit");
        let _ = self.process.wait();
        Ok(())
    }
}

// ── Battle ───────────────────────────────────────────────────────────────────

struct Battle {
    game: TicTacToe,
    engines: [Engine; 2], // [0] = X (moves first), [1] = O
    movetime_ms: u64,     // kept for future use / logging
    ply_index: usize,
    last_move: Option<u8>,
}

impl Battle {
    fn new(path1: &str, path2: &str) -> Result<Self> {
        let e1 = Engine::spawn(path1)?;
        let e2 = Engine::spawn(path2)?;
        println!("Engine X: {}", e1.name);
        println!("Engine O: {}", e2.name);
        Ok(Self {
            game: TicTacToe::new(),
            engines: [e1, e2],
            movetime_ms: 1000,
            ply_index: 0,
            last_move: None,
        })
    }

    fn start(&mut self) -> Result<()> {
        self.engines[0].new_game(true)?;
        self.engines[1].new_game(false)?;

        println!("{}", self.game);

        // X moves first without receiving an opponent square
        let first = self.engines[0].go_first()?;
        if let Err(e) = self.apply(first, 0)? {
            return Ok(e);
        }

        loop {
            let turn = self.ply_index % 2; // 0 = X, 1 = O
            let _other = 1 - turn;
            let last = self.last_move.expect("at least one move played");

            // send opponent's last square → get this engine's reply
            let cell = self.engines[turn].send_opponent_move(last)?;

            if let Err(e) = self.apply(cell, turn)? {
                return Ok(e);
            }
        }
    }

    /// Validate and apply `cell` played by `turn` (0=X,1=O).
    /// Returns Ok(Ok(())) to continue, Ok(Err(msg)) to stop cleanly.
    fn apply(&mut self, cell: u8, turn: usize) -> Result<Result<(), ()>> {
        println!(
            "Turn {} — {} ('{}') plays {cell}",
            self.ply_index + 1,
            if turn == 0 { "X" } else { "O" },
            self.engines[turn].name,
        );

        match self.game.validate_move(cell) {
            Ok(_) => {
                self.game.make(cell);
                self.last_move = Some(cell);
                self.ply_index += 1;
            }
            Err(e) => {
                println!(
                    "Illegal move by '{}': {e} — forfeit.",
                    self.engines[turn].name
                );
                println!("'{}' wins.", self.engines[1 - turn].name);
                return Ok(Err(()));
            }
        }

        println!("{}", self.game);

        if self.game.check_win() {
            println!(
                "Game over — '{}' wins after {} moves.",
                self.engines[turn].name, self.ply_index,
            );
            return Ok(Err(()));
        }
        if self.game.check_draw() {
            println!("Game over — draw after {} moves.", self.ply_index);
            return Ok(Err(()));
        }

        Ok(Ok(()))
    }
}

// ── Drop: always kill child processes ────────────────────────────────────────

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.process.kill();
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        bail!("usage: {} <engine1> <engine2> [movetime_ms]", args[0]);
    }
    let mut battle = Battle::new(&args[1], &args[2])?;
    if let Some(ms) = args.get(3) {
        battle.movetime_ms = ms.parse().context("movetime_ms must be an integer")?;
    }
    battle.start()?;

    // ensure engines are quit cleanly before Drop kills them
    battle.engines[0].quit()?;
    battle.engines[1].quit()?;
    Ok(())
}
