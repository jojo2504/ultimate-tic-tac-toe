use crate::constants::{
    CELL_TO_SUBBOARD_BASE, CELL_TO_SUBBOARD_FOCUS, CELL_TO_SUBBOARD_INDEX, CHECKERS,
    FEATURES_COUNT, FINAL_CHECKERS, MAP, WINDOW,
};
use colored::Colorize;
use once_cell::sync::Lazy;
use rand::random;
use std::fmt;

#[derive(Debug, Default, Clone, Copy)]
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

#[derive(Debug)]
pub enum Result {
    Win,
    Loss,
    Draw,
}

pub struct MoveDelta {
    pub square: u8,
    pub turn: Symbol,
    pub new_focus: Option<u8>,
    pub cleared_board: Option<u8>, // if a board was just won/filled
}

#[derive(Default, Clone, Copy)]
pub struct TicTacToe {
    // The first 47 (128 - 91) bits will never be used on these bitboards
    pub bitboard: u128,
    pub side_bitboard: u128,
    pub zobrist_key: u128,

    pub all_clear: u16,     // 0 if not cleared, else 1
    pub side_clear: u16,    // 1s for each sub board cleared by white and black alternatively
    pub full_subboard: u16, // tracks the subboards which are full, need to separated all_clear and this else risk of corruption of side_clear

    pub current_focus: Option<u8>, // the forced board to play on, None if impossible giving a free board focus
    pub turn: Symbol,
    pub ply: usize,
}

impl TicTacToe {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn validate_move(&self, square: u8) -> anyhow::Result<()> {
        if self.all_clear & (1 << CELL_TO_SUBBOARD_INDEX[square as usize]) == 0
            && (self.bitboard & (1 << square)) == 0
            && (self.current_focus.is_none()
                || CELL_TO_SUBBOARD_INDEX[square as usize] == self.current_focus.unwrap())
        {
            return Ok(());
        }
        anyhow::bail!("invalid move");
    }

    /// Suppose move has already been validated
    pub fn make(&mut self, square: u8) -> MoveDelta {
        let old_clear = self.all_clear;

        self.bitboard ^= 1 << square;
        self.side_bitboard ^= self.bitboard;

        self.zobrist_key ^=
            ZOBRIST_TABLE.token_square[Zobrist::get_index((square, self.turn.clone()))];

        if let Some(board_index) = self.check_board_clear(square) {
            self.all_clear ^= 1 << board_index;
        }
        self.side_clear ^= self.all_clear;

        self.current_focus =
            ((1u16 << CELL_TO_SUBBOARD_FOCUS[square as usize]) & self.all_clear as u16 == 0)
                .then_some(CELL_TO_SUBBOARD_FOCUS[square as usize]);

        let cleared_board = if self.all_clear != old_clear {
            Some((self.all_clear ^ old_clear).trailing_zeros() as u8)
        } else {
            None
        };

        let delta = MoveDelta {
            square,
            turn: self.turn,
            new_focus: self.current_focus,
            cleared_board,
        };

        self.turn = self.turn.swap();
        self.ply += 1;

        delta
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
            self.full_subboard ^= 1 << CELL_TO_SUBBOARD_INDEX[square as usize];
        }

        None
    }

    /// Used as a helper after a make move\
    /// Returns true if a player cleared 3 aligned boards.
    pub fn check_win(&self) -> bool {
        let mask = &self.side_clear;
        FINAL_CHECKERS
            .iter()
            .any(|checker| mask & checker == *checker)
    }

    #[inline(always)]
    pub fn check_draw(&self) -> bool {
        !self.check_win() && self.is_full()
    }

    pub fn is_game_over(&self) -> bool {
        self.check_win() || self.check_draw()
    }

    // give the result from the current player perspective
    pub fn result(&self) -> Result {
        if self.check_win() {
            return Result::Win;
        } else if self.check_draw() {
            return Result::Draw;
        }
        return Result::Loss;
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        (self.full_subboard | self.all_clear) == 0b111111111
    }

    /// Returns features.
    ///
    /// XOR-toggle invariant: `side_bitboard` always holds the opponent's
    /// pieces, `side_bitboard ^ bitboard` always holds the current
    /// player's pieces — regardless of which side is to move.
    pub fn to_features(&self) -> [f32; FEATURES_COUNT] {
        let mut features = [0.0f32; FEATURES_COUNT];

        let (cross_bb, circle_bb, cross_clear, circle_clear) = match self.turn as i32 {
            1 => (
                self.side_bitboard ^ self.bitboard,
                self.side_bitboard,
                self.side_clear ^ (self.all_clear as u16),
                self.side_clear,
            ),
            -1 => (
                self.side_bitboard,
                self.side_bitboard ^ self.bitboard,
                self.side_clear,
                self.side_clear ^ (self.all_clear as u16),
            ),
            _ => unreachable!(),
        };

        // 81 bits — current player raw bitboard
        for i in 0..81 {
            features[i] = ((cross_bb >> i) & 1) as f32;
        }

        // 81 bits — opponent raw bitboard
        for i in 0..81 {
            features[81 + i] = ((circle_bb >> i) & 1) as f32;
        }

        // 9 bits — current player cleared sub-boards (meta board)
        for i in 0..9 {
            features[162 + i] = ((cross_clear >> i) & 1) as f32;
        }

        // 9 bits — opponent cleared sub-boards (meta board)
        for i in 0..9 {
            features[171 + i] = ((circle_clear >> i) & 1) as f32;
        }

        // 9 bits — all_clear (dead/drawn sub-boards)
        for i in 0..9 {
            features[180 + i] = ((self.all_clear >> i) & 1) as f32;
        }

        // 10 bits — current_focus as one-hot + free choice flag
        match self.current_focus {
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
        let (cross, circle): (u128, u128) = match self.turn as i32 {
            1 => (self.side_bitboard ^ self.bitboard, self.side_bitboard),
            -1 => (self.side_bitboard, self.side_bitboard ^ self.bitboard),
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
