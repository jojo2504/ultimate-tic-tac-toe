use crate::constants::{
    self, CELL_TO_SUBBOARD_BASE, CELL_TO_SUBBOARD_FOCUS, CELL_TO_SUBBOARD_INDEX, CHECKERS,
    FEATURES_COUNT, FINAL_CHECKERS, MAP, WINDOW,
};
use colored::Colorize;
use once_cell::sync::Lazy;
use rand::random;
use std::{any, fmt};

#[derive(Debug, Default, Clone)]
pub enum Symbol {
    #[default]
    Cross = 1,
    Circle = -1,
}

impl Symbol {
    pub fn swap(&self) -> Self {
        match self {
            Symbol::Cross => Symbol::Circle,
            Symbol::Circle => Symbol::Cross,
        }
    }
}

impl From<Symbol> for i32 {
    fn from(s: Symbol) -> i32 {
        s as i32
    }
}

#[derive(Default)]
pub struct State {
    pub all_clear: u16,  // 0 if not cleared, else 1
    pub side_clear: u16, // 1s for each sub board cleared by white and black alternatively
    pub turn: Symbol,
    pub last_move: Option<u8>,     // index of the nth bits played
    pub current_focus: Option<u8>, // the forced board to play on, None if impossible giving a free board focus
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Undo {
    pub all_clear: u16, // 0 if not cleared, else 1
    pub side_clear: u16,
    pub current_focus: Option<u8>, // index of the focused bitboard
}

/// The first 47 (128 - 91) bits will never be used on this bitboard
pub struct TicTacToe {
    pub bitboard: u128,
    pub side_bitboard: u128,
    pub state: State,
    /// Used to keep track of all undo needed to restore the state in the unmake function.
    pub undo_stack: Box<[Undo; 81]>,
    /// Used to index the state_stack, representing the current ply, equivalent to a half-move.
    pub ply_index: usize,
    pub zobrist_key: u128,
    pub winner: Option<Symbol>,
}

impl TicTacToe {
    pub fn new() -> Self {
        Self {
            bitboard: 0,
            side_bitboard: 0,
            state: State::default(),
            undo_stack: Box::new([Undo::default(); 81]),
            ply_index: 0,
            zobrist_key: 0,
            winner: None,
        }
    }

    pub fn validate_move(&self, square: u8) -> anyhow::Result<()> {
        if self.state.all_clear & (1 << CELL_TO_SUBBOARD_INDEX[square as usize]) == 0
            && (self.bitboard & (1 << square)) == 0
        {
            return Ok(());
        }
        anyhow::bail!("invalid move");
    }

    /// Suppose move has already been validated
    #[inline(always)]
    pub fn make(&mut self, square: u8) {
        let undo = Undo {
            all_clear: self.state.all_clear,
            side_clear: self.state.side_clear,
            current_focus: self.state.current_focus,
        };
        self.undo_stack[self.ply_index] = undo;
        self.play(square);
        self.state.turn = self.state.turn.swap();
        self.ply_index += 1;
    }

    fn play(&mut self, square: u8) {
        self.state.last_move = Some(square);

        self.bitboard ^= 1 << square;
        self.side_bitboard ^= self.bitboard;

        self.zobrist_key ^=
            ZOBRIST_TABLE.token_square[Zobrist::get_index((square, self.state.turn.clone()))];

        if let Some(board_index) = self.check_board_clear(square) {
            self.state.all_clear ^= 1 << board_index;
        }
        self.state.side_clear ^= self.state.all_clear;

        self.state.current_focus =
            ((1u16 << CELL_TO_SUBBOARD_FOCUS[square as usize]) & self.state.all_clear as u16 == 0)
                .then_some(CELL_TO_SUBBOARD_FOCUS[square as usize]);
    }

    pub fn unmake(&mut self, square: u8) {
        self.ply_index -= 1;
        let undo = self.undo_stack[self.ply_index];
        self.state.turn = self.state.turn.swap();

        // revert the states with the undo
        self.state.all_clear = undo.all_clear;
        self.state.side_clear = undo.side_clear;
        self.state.current_focus = undo.current_focus;

        self.bitboard ^= 1 << square;
        self.side_bitboard ^= self.bitboard;
        self.zobrist_key ^=
            ZOBRIST_TABLE.token_square[Zobrist::get_index((square, self.state.turn.clone()))];

        self.winner = None;
    }

    /// Used as a helper during a make move\
    /// Returns the nth bit 0 to 8 of the current cleared board.\
    /// Doesn't return a bitboard.
    fn check_board_clear(&mut self, square: u8) -> Option<u8> {
        let base = CELL_TO_SUBBOARD_BASE[square as usize];
        let mask = &self.side_bitboard;
        if CHECKERS
            .iter()
            .any(|checker| mask & (checker << base) == (checker << base))
        {
            return Some(CELL_TO_SUBBOARD_INDEX[square as usize]);
        }

        let mask = WINDOW << MAP[CELL_TO_SUBBOARD_INDEX[square as usize] as usize];
        if mask & self.bitboard == mask {
            self.state.all_clear ^= 1 << CELL_TO_SUBBOARD_INDEX[square as usize];
        }

        None
    }

    /// Used as a helper after a make move\
    /// Returns true if a player cleared 3 aligned boards.
    pub fn check_win(&self) -> bool {
        let mask = &self.state.side_clear;
        FINAL_CHECKERS
            .iter()
            .any(|checker| mask & checker == *checker)
    }

    pub fn check_draw(&self) -> bool {
        self.winner.is_none() && self.is_full()
    }

    pub fn is_full(&self) -> bool {
        self.state.all_clear == 0b111111111
    }

    pub fn to_features(&self) -> [f32; FEATURES_COUNT] {
        let mut features = [0.0f32; FEATURES_COUNT];

        // Determine which bitboard belongs to whom based on ply parity
        let (current_bb, opponent_bb, current_clear, opponent_clear) = match self.ply_index & 1 {
            0 => (
                self.side_bitboard,
                self.side_bitboard ^ self.bitboard,
                self.state.side_clear,
                self.state.side_clear ^ (self.state.all_clear as u16),
            ),
            1 => (
                self.side_bitboard ^ self.bitboard,
                self.side_bitboard,
                self.state.side_clear ^ (self.state.all_clear as u16),
                self.state.side_clear,
            ),
            _ => unreachable!(),
        };

        // 81 bits — current player raw bitboard
        for i in 0..81 {
            features[i] = ((current_bb >> i) & 1) as f32;
        }

        // 81 bits — opponent raw bitboard
        for i in 0..81 {
            features[81 + i] = ((opponent_bb >> i) & 1) as f32;
        }

        // 9 bits — current player cleared sub-boards (meta board)
        for i in 0..9 {
            features[162 + i] = ((current_clear >> i) & 1) as f32;
        }

        // 9 bits — opponent cleared sub-boards (meta board)
        for i in 0..9 {
            features[171 + i] = ((opponent_clear >> i) & 1) as f32;
        }

        // 9 bits — all_clear (dead/drawn sub-boards)
        for i in 0..9 {
            features[180 + i] = ((self.state.all_clear >> i) & 1) as f32;
        }

        // 10 bits — current_focus as one-hot + free choice flag
        match self.state.current_focus {
            Some(focus) => {
                features[189 + focus as usize] = 1.0; // one-hot
                features[198] = 0.0; // not free
            }
            None => {
                // all focus bits stay 0, free choice flag = 1
                features[198] = 1.0;
            }
        }

        features
    }
}

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (circle, cross): (u128, u128) = match self.ply_index & 1 {
            0 => (self.side_bitboard, self.side_bitboard ^ self.bitboard),
            1 => (self.side_bitboard ^ self.bitboard, self.side_bitboard),
            _ => unreachable!(),
        };

        writeln!(f, "{}", "      0  1  2   3  4  5   6  7  8".dimmed())?;
        writeln!(f, "{}", "    ┌─────────┬─────────┬─────────┐".dimmed())?;

        for row in 0..9 {
            let row_start = row * 9;
            write!(f, "{}", format!("{:2}  │", row_start).dimmed())?;

            for col in 0..9 {
                let bit = row_start + col;
                let mask = 1u128 << bit;
                let cell = if cross & mask != 0 {
                    " X ".white().on_blue().bold()
                } else if circle & mask != 0 {
                    " O ".black().on_red().bold()
                } else {
                    " . ".dimmed()
                };
                write!(f, "{}", cell)?;
                if col == 2 || col == 5 {
                    write!(f, "{}", "│".dimmed())?;
                }
            }
            write!(f, "{}", "│".dimmed())?;
            writeln!(f)?;

            if row == 2 || row == 5 {
                writeln!(f, "{}", "    ├─────────┼─────────┼─────────┤".dimmed())?;
            }
        }
        writeln!(f, "{}", "    └─────────┴─────────┴─────────┘".dimmed())?;
        Ok(())
    }
}

struct Zobrist {
    token_square: [u128; 81 * 2], // 42 * 2
}

impl Default for Zobrist {
    fn default() -> Self {
        Self {
            token_square: [0u128; 81 * 2],
        }
    }
}

static ZOBRIST_TABLE: Lazy<Zobrist> = Lazy::new(|| {
    let mut z = Zobrist::default();
    for i in 0..z.token_square.len() {
        z.token_square[i] = random();
    }
    z
});

impl Zobrist {
    fn get_index(play: (u8, Symbol)) -> usize {
        let offset = match play.1.clone() as i32 {
            1 => 0,
            -1 => 1,
            _ => unreachable!(),
        };

        (offset as u8 * 81 + play.0) as usize
    }
}
