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
    pub square: u8,                // 0 - 80
    pub turn: Symbol,              // who made this move (before swap)
    pub old_focus: Option<u8>,     // FIX B: focus BEFORE the move (needed for subtraction)
    pub new_focus: Option<u8>,     // focus AFTER the move
    pub cleared_board: Option<u8>, // 0 - 8 if a sub-board was won/drawn this move
}

#[derive(Default, Clone, Copy)]
pub struct TicTacToe {
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

    /// Suppose move has already been validated.
    pub fn make(&mut self, square: u8) -> MoveDelta {
        let old_clear = self.all_clear;
        let old_focus = self.current_focus;

        self.bitboard ^= 1 << square;
        self.side_bitboard ^= self.bitboard;

        self.zobrist_key ^=
            ZOBRIST_TABLE.token_square[Zobrist::get_index((square, self.turn.clone()))];

        if let Some(board_index) = self.check_board_clear(square) {
            self.all_clear ^= 1 << board_index;
            self.zobrist_key ^= ZOBRIST_TABLE.token_all_clear[board_index as usize];
        }
        self.side_clear ^= self.all_clear;

        match old_focus {
            Some(f) => self.zobrist_key ^= ZOBRIST_TABLE.token_focus[f as usize],
            None => self.zobrist_key ^= ZOBRIST_TABLE.token_focus[9],
        }

        self.current_focus =
            ((1u16 << CELL_TO_SUBBOARD_FOCUS[square as usize]) & self.all_clear as u16 == 0)
                .then_some(CELL_TO_SUBBOARD_FOCUS[square as usize]);

        match self.current_focus {
            Some(f) => self.zobrist_key ^= ZOBRIST_TABLE.token_focus[f as usize],
            None => self.zobrist_key ^= ZOBRIST_TABLE.token_focus[9],
        }

        let cleared_board = if self.all_clear != old_clear {
            Some((self.all_clear ^ old_clear).trailing_zeros() as u8)
        } else {
            None
        };

        let delta = MoveDelta {
            square,
            turn: self.turn,
            old_focus, // FIX B
            new_focus: self.current_focus,
            cleared_board,
        };

        self.turn = self.turn.swap();
        self.ply += 1;

        delta
    }

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

    /// Features are always laid out as (STM = side to move, NSTM = opponent):
    ///   [  0.. 80] STM  piece squares     (81 bits)
    ///   [ 81..161] NSTM piece squares     (81 bits)
    ///   [162..170] STM  cleared meta-board (9 bits)
    ///   [171..179] NSTM cleared meta-board (9 bits)
    ///   [180..188] all_clear (symmetric)   (9 bits)
    ///   [189..197] current_focus one-hot   (9 bits)
    ///   [198]      free-choice flag
    ///
    /// Previously, indices 0..81 were always Cross regardless of whose turn it was,
    /// causing the network to see identical inputs for contradictory targets.
    pub fn to_features(&self) -> [f32; FEATURES_COUNT] {
        let mut features = [0.0f32; FEATURES_COUNT];

        // Bitboard invariant after make():
        //   side_bitboard           = NSTM pieces
        //   side_bitboard ^ bitboard = STM  pieces
        let stm_bb = self.side_bitboard ^ self.bitboard;
        let nstm_bb = self.side_bitboard;

        // Cleared-board invariant (proved by tracing make()):
        //   stm_clear  = side_clear ^ all_clear
        //   nstm_clear = side_clear
        let stm_clear = self.side_clear ^ self.all_clear as u16;
        let nstm_clear = self.side_clear;

        for i in 0..81 {
            features[i] = ((stm_bb >> i) & 1) as f32;
            features[81 + i] = ((nstm_bb >> i) & 1) as f32;
        }
        for i in 0..9 {
            features[162 + i] = ((stm_clear >> i) & 1) as f32;
            features[171 + i] = ((nstm_clear >> i) & 1) as f32;
            features[180 + i] = ((self.all_clear >> i) & 1) as f32;
        }
        match self.current_focus {
            Some(f) => {
                features[189 + f as usize] = 1.0;
            }
            None => {
                features[198] = 1.0;
            }
        }

        features
    }
}

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stm_bb = self.side_bitboard ^ self.bitboard;
        let nstm_bb = self.side_bitboard;
        let (cross, circle) = match self.turn as i32 {
            1 => (stm_bb, nstm_bb),
            -1 => (nstm_bb, stm_bb),
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
    token_square: [u128; 81 * 2],
    token_focus: [u128; 10], // [0..8] board focus, [9] free-choice
    token_all_clear: [u128; 9],
}

impl Default for Zobrist {
    fn default() -> Self {
        Self {
            token_square: [0u128; 81 * 2],
            token_focus: [0u128; 10],
            token_all_clear: [0u128; 9],
        }
    }
}

static ZOBRIST_TABLE: Lazy<Zobrist> = Lazy::new(|| {
    let mut z = Zobrist::default();
    for v in z.token_square.iter_mut() {
        *v = random();
    }
    for v in z.token_focus.iter_mut() {
        *v = random();
    }
    for v in z.token_all_clear.iter_mut() {
        *v = random();
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
