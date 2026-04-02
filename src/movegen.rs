use rand::random_range;

use crate::core::{CELL_TO_SUBBOARD_INDEX, MAP, TicTacToe, WINDOW};

pub fn generate_random_legal_move(game: &TicTacToe) -> u8 {
    let moves: u128 = generate_moves(&game);

    let pick = random_range(0..moves.count_ones());
    // skip `pick` set bits
    let mut remaining = moves;
    for _ in 0..pick {
        remaining &= remaining - 1; // clear lowest set bit
    }
    let random_move = 1u128 << remaining.trailing_zeros();
    let random_move_index = random_move.trailing_zeros() as u8;
    random_move_index
}

/// Generate all moves and return them into another bitboard.
pub fn generate_moves(board: &TicTacToe) -> u128 {
    let occupied = board.bitboard;

    let mask = if let Some(current_focus) = board.state.current_focus
        && (board.state.all_clear & (1 << board.state.current_focus.unwrap())) != 0
    {
        WINDOW << MAP[current_focus as usize]
    } else {
        (0..81u8)
            .filter(|&i| {
                (1 << CELL_TO_SUBBOARD_INDEX[i as usize]) as u16 & board.state.all_clear == 0
            })
            .fold(0u128, |acc, i| acc | (1 << i))
    };

    mask & !occupied
}
