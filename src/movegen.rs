use crate::core::TicTacToe;

const WINDOW: u128 = 0b000000111000000111000000111;

/// Generate all moves and return them into another bitboard.
pub fn generate_moves(board: &TicTacToe) -> u128 {
    let occupied = board.bitboard;

    let mask = if let Some(current_focus) = board.state.current_focus {
        WINDOW << current_focus
    } else {
        (0..9u8)
            .filter(|&i| (1 << i) & board.state.all_clear == 0)
            .fold(0u128, |acc, i| acc | (WINDOW << i))
    };

    mask & !occupied
}
