pub const MAP: [u8; 9] = [0, 3, 6, 27, 30, 33, 54, 57, 60];
pub const WINDOW: u128 = 0b000000111000000111000000111; // the top left sub board

/// Map the cell index to a subboard index\
/// Example:\
/// 1 -> 0, 10 -> 1, 4 -> 2
pub const CELL_TO_SUBBOARD_INDEX: [u8; 81] = {
    let mut arr = [0u8; 81];
    let mut i = 0;
    while i < 9 {
        let base = MAP[i] as usize;
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
pub const CELL_TO_SUBBOARD_BASE: [u8; 81] = {
    let mut arr = [0u8; 81];
    let mut i = 0;
    while i < 9 {
        let base = MAP[i] as usize;
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

/// Map the cell index to a subboard for focusing
pub const CELL_TO_SUBBOARD_FOCUS: [u8; 81] = {
    let mut arr = [0u8; 81];
    let mut i = 0;
    while i < 9 {
        let base = MAP[i] as usize;
        arr[base] = 0 as u8;
        arr[base + 1] = 1 as u8;
        arr[base + 2] = 2 as u8;
        arr[base + 9] = 3 as u8;
        arr[base + 10] = 4 as u8;
        arr[base + 11] = 5 as u8;
        arr[base + 18] = 6 as u8;
        arr[base + 19] = 7 as u8;
        arr[base + 20] = 8 as u8;
        i += 1;
    }
    arr
};

// Need to be at least u128 to contains 81 bits
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

// Need to be at least u16 to contains 9 bits
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

pub const FEATURES_COUNT: usize = 199;
