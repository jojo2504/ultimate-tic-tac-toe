use std::{fmt, u128};

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

#[derive(Debug, Copy, Clone, Default)]
pub struct Undo {
    white_clear: u8, // 1s for each sub board cleared by white
    black_clear: u8, // 1s for each sub board cleared by black
    current_focus: Option<u8>,
}

#[derive(Default)]
pub struct State {
    white_clear: u8, // 1s for each sub board cleared by white
    black_clear: u8, // 1s for each sub board cleared by black
    player_turn: Color,
    last_move: Option<u8>,     // index of the nth bits played
    current_focus: Option<u8>, // the forced board to play on, None if impossible giving a free board focus
}

/// The first 47 (128 - 91) bits will never be used on this bitboard
pub struct TicTacToe {
    bitboard: u128,
    white_bitboard: u128,
    black_bitboard: u128,
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

        match self.state.player_turn {
            Color::White => self.white_bitboard ^= 1 << square,
            Color::Black => self.black_bitboard ^= 1 << square,
        }
        self.bitboard ^= 1 << square;
        self.state.player_turn = self.state.player_turn.swap();
    }

    pub fn unmake(&mut self, square: u8) {
        todo!()
    }

    /// used as a helper after a make move
    fn check_board_clear(&self) -> bool {
        todo!()
    }

    /// used as a helper after a make move
    pub fn check_win(&self) -> bool {
        todo!()
    }
}

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}
