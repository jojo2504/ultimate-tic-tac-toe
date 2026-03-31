use crate::core::{CELL_TO_SUBBOARD_INDEX, MAP, TicTacToe, WINDOW};

/// Generate all moves and return them into another bitboard.
pub fn generate_moves(board: &TicTacToe) -> u128 {
    let occupied = board.bitboard;

    let mask = if let Some(current_focus) = board.state.current_focus {
        WINDOW << MAP[current_focus as usize]
    } else {
        (0..81u8)
            .filter(|&i| {
                (1 << CELL_TO_SUBBOARD_INDEX[i as usize]) as u16 & board.state.all_clear == 0
            })
            .fold(0u128, |acc, i| acc | (1 << i))
    };

    // println!("{:09b}", board.state.all_clear);
    // println!("{:081b}", mask);
    // println!("{:081b}", occupied);
    // println!("{:081b}", mask & !occupied);

    mask & !occupied
}
