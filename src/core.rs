use std::fmt;

use colored::Colorize;

/// Map the cell index to a subboard index
pub const CELL_TO_SUBBOARD_INDEX: [u8; 81] = {
    let mut arr = [0u8; 81];
    let map: [u8; 9] = [0, 3, 6, 27, 30, 33, 54, 57, 60];
    let mut i = 0;
    while i < 9 {
        let base = map[i] as usize;
        arr[base] = i as u8;
        arr[base + 1] = i as u8;
        arr[base + 2] = i as u8;
        arr[base + 9] = i as u8;
        arr[base + 10] = i as u8;
        arr[base + 11] = i as u8;
        arr[base + 18] = i as u8;
        arr[base + 19] = i as u8;
        arr[base + 20] = i as u8;
        i += 1;
    }
    arr
};

/// Map the cell index to a subboard first square index
///
/// Example:\
/// 1 -> 0, 10 -> 1, 4 -> 2
pub const CELL_TO_SUBBOARD_BASE: [u8; 81] = {
    let mut arr = [0u8; 81];
    let map: [u8; 9] = [0, 3, 6, 27, 30, 33, 54, 57, 60];
    let mut i = 0;
    while i < 9 {
        let base = map[i] as usize;
        arr[base] = base as u8;
        arr[base + 1] = base as u8;
        arr[base + 2] = base as u8;
        arr[base + 9] = base as u8;
        arr[base + 10] = base as u8;
        arr[base + 11] = base as u8;
        arr[base + 18] = base as u8;
        arr[base + 19] = base as u8;
        arr[base + 20] = base as u8;
        i += 1;
    }
    arr
};

/// ```
/// let h1 = 0b111u128;
/// let h2 = 0b111000000000u128;
/// let h3 = 0b111000000000000000000u128;
/// let v1 = 0b1000000001000000001u128;
/// let v2 = 0b10000000010000000010u128;
/// let v3 = 0b100000000100000000100u128;
/// let diag = 0b100000000010000000001u128;
/// let anti_diag = 0b1000000010000000100u128;
/// /// ```
pub const CHECKERS: [u128; 8] = [
    0b111u128,
    0b111000000000u128,
    0b111000000000000000000u128,
    0b1000000001000000001u128,
    0b10000000010000000010u128,
    0b100000000100000000100u128,
    0b100000000010000000001u128,
    0b1000000010000000100u128,
];

/// ```
/// let h1 = 0b111u128;
/// let h2 = 0b111000000000u128;
/// let h3 = 0b111000000000000000000u128;
/// let v1 = 0b1000000001000000001u128;
/// let v2 = 0b10000000010000000010u128;
/// let v3 = 0b100000000100000000100u128;
/// let diag = 0b100000000010000000001u128;
/// let anti_diag = 0b1000000010000000100u128;
/// ```
/// Need to be at least u16 to contains 9 bits
pub const FINAL_CHECKERS: [u16; 8] = [
    0b111u16,
    0b111000u16,
    0b111000000u16,
    0b1001001u16,
    0b10010010u16,
    0b100100100u16,
    0b100010001u16,
    0b1010100u16,
];

#[derive(Debug, Default)]
pub enum Color {
    #[default]
    White,
    Black,
}

impl Color {
    pub fn swap(&self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Default)]
pub struct State {
    pub all_clear: u8,   // 0 if not cleared, else 1
    pub white_clear: u8, // 1s for each sub board cleared by white
    pub black_clear: u8, // 1s for each sub board cleared by black
    pub player_turn: Color,
    pub last_move: Option<u8>,     // index of the nth bits played
    pub current_focus: Option<u8>, // the forced board to play on, None if impossible giving a free board focus
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Undo {
    pub all_clear: u8,   // 0 if not cleared, else 1
    pub white_clear: u8, // 1s for each sub board cleared by white
    pub black_clear: u8, // 1s for each sub board cleared by black
    pub current_focus: Option<u8>,
}

/// The first 47 (128 - 91) bits will never be used on this bitboard
pub struct TicTacToe {
    pub bitboard: u128,
    pub white_bitboard: u128,
    pub black_bitboard: u128,
    pub state: State,
    /// Used to keep track of all undo needed to restore the state in the unmake function.
    pub undo_stack: Box<[Undo; 5096]>,
    /// Used to index the state_stack, representing the current ply, equivalent to a half-move.
    pub ply_index: usize,
}

impl TicTacToe {
    pub fn new() -> Self {
        Self {
            bitboard: 0,
            white_bitboard: 0,
            black_bitboard: 0,
            state: State::default(),
            undo_stack: Box::new([Undo::default(); 5096]),
            ply_index: 0,
        }
    }

    pub fn make(&mut self, square: u8) {
        self.state.last_move = Some(square);

        let undo = Undo {
            all_clear: self.state.all_clear,
            white_clear: self.state.white_clear,
            black_clear: self.state.black_clear,
            current_focus: self.state.current_focus,
        };

        match self.state.player_turn {
            Color::White => self.white_bitboard ^= 1 << square,
            Color::Black => self.black_bitboard ^= 1 << square,
        }
        self.bitboard ^= 1 << square;

        if let Some(bit_index) = self.check_board_clear(square) {
            match self.state.player_turn {
                Color::White => self.state.white_clear ^= bit_index,
                Color::Black => self.state.black_clear ^= bit_index,
            }
            self.state.all_clear ^= bit_index;
        }

        self.state.player_turn = self.state.player_turn.swap();
        self.undo_stack[self.ply_index] = undo;
        self.ply_index += 1;
    }

    pub fn unmake(&mut self, square: u8) {
        self.ply_index -= 1;
        let undo = self.undo_stack[self.ply_index];
        self.state.player_turn = self.state.player_turn.swap();

        // revert the states with the undo
        self.state.all_clear = undo.all_clear;
        self.state.black_clear = undo.black_clear;
        self.state.white_clear = undo.white_clear;
        self.state.current_focus = undo.current_focus;

        match self.state.player_turn {
            Color::White => self.white_bitboard ^= 1 << square,
            Color::Black => self.black_bitboard ^= 1 << square,
        }
        self.bitboard ^= 1 << square;
    }

    /// Used as a helper during a make move\
    /// Returns the nth bit 0 to 8 of the current cleared board.
    fn check_board_clear(&self, square: u8) -> Option<u8> {
        let base = CELL_TO_SUBBOARD_BASE[square as usize];
        let mask = match self.state.player_turn {
            Color::White => self.white_bitboard,
            Color::Black => self.black_bitboard,
        };
        if CHECKERS
            .iter()
            .any(|checker| mask & (checker << base) == (checker << base))
        {
            return Some(square);
        }
        None
    }

    /// Used as a helper after a make move\
    /// Returns true if a player cleared 3 aligned boards.
    pub fn check_win(&self) -> bool {
        let mask = match self.state.player_turn {
            Color::White => self.state.white_clear,
            Color::Black => self.state.black_clear,
        } as u16;
        FINAL_CHECKERS
            .iter()
            .any(|checker| mask & checker == *checker)
    }
}

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..9 {
            for col in 0..9 {
                let bit = row * 9 + col;
                let mask = 1u128 << bit;

                let cell = if self.white_bitboard & mask != 0 {
                    " X ".white().on_blue().bold()
                } else if self.black_bitboard & mask != 0 {
                    " O ".black().on_red().bold()
                } else {
                    " . ".dimmed()
                };

                write!(f, "{}", cell)?;

                // vertical separator between 3x3 boxes
                if col == 2 || col == 5 {
                    write!(f, "{}", "|".dimmed())?;
                }
            }
            writeln!(f)?;

            // horizontal separator between 3x3 boxes
            if row == 2 || row == 5 {
                writeln!(f, "{}", "- - - - - - - - - - - - - - -".dimmed())?;
            }
        }
        Ok(())
    }
}
