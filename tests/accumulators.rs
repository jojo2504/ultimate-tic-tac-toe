#[cfg(test)]
mod tests {
    use ultimate_tic_tac_toe::{
        core::{Symbol, TicTacToe},
        network::{DualAccumulator, Network, get_bucket},
        search::Search,
    };

    fn load_net() -> Box<Network> {
        Network::load("databin/gen0_weights.bin".to_owned())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a DualAccumulator from scratch for `board`, then apply all moves
    /// described by `squares` incrementally.  Returns the result.
    fn build_incremental(net: &Network, squares: &[u8]) -> DualAccumulator {
        let mut board = TicTacToe::new();
        // Start from a scratch accumulator for the empty board
        let mut acc = DualAccumulator::new(net, &board);
        for &sq in squares {
            let delta = board.make(sq);
            acc.apply_delta(net, &delta);
        }
        acc
    }

    /// Build a DualAccumulator from scratch after applying all moves to a board.
    fn build_from_scratch(net: &Network, squares: &[u8]) -> DualAccumulator {
        let mut board = TicTacToe::new();
        for &sq in squares {
            board.make(sq);
        }
        DualAccumulator::new(net, &board)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Incrementally-updated accumulator must equal from-scratch after 0 moves.
    #[test]
    fn accumulator_empty_board() {
        let net = load_net();
        let board = TicTacToe::new();
        let scratch = DualAccumulator::new(&net, &board);
        let incremental = DualAccumulator::new(&net, &board);
        assert_eq!(scratch.acc[0], incremental.acc[0]);
        assert_eq!(scratch.acc[1], incremental.acc[1]);
    }

    /// After one move the incremental acc must match the from-scratch acc.
    #[test]
    fn accumulator_one_move() {
        let net = load_net();
        let moves = [0u8];
        let inc = build_incremental(&net, &moves);
        let scr = build_from_scratch(&net, &moves);
        assert_eq!(inc.acc[0], scr.acc[0], "acc[0] mismatch after 1 move");
        assert_eq!(inc.acc[1], scr.acc[1], "acc[1] mismatch after 1 move");
    }

    /// After two moves (one per side).
    #[test]
    fn accumulator_two_moves() {
        let net = load_net();
        // Square 0 sends focus to board 0; second move must be in board 0 → square 0 is taken,
        // pick square 1 (still in board 0 since focus = board at col 0 = board 0? no).
        // Actually CELL_TO_SUBBOARD_FOCUS[0] = 0, so focus goes to board 0.
        // Board 0 squares via MAP[0]=0: 0,1,2,9,10,11,18,19,20.  Pick 1.
        let moves = [0u8, 1u8];
        let inc = build_incremental(&net, &moves);
        let scr = build_from_scratch(&net, &moves);
        assert_eq!(inc.acc[0], scr.acc[0], "acc[0] mismatch after 2 moves");
        assert_eq!(inc.acc[1], scr.acc[1], "acc[1] mismatch after 2 moves");
    }

    /// Six plies deep — exercises focus subtraction and the all_clear path.
    #[test]
    fn accumulator_six_moves() {
        let net = load_net();
        // A sequence that stays legal: each move's destination board is not full/cleared.
        // CELL_TO_SUBBOARD_FOCUS[sq] determines the next focus.
        // sq 0 → focus board 0; pick sq in board 0 next: sq 1 (focus→board 1); etc.
        let moves = [0u8, 1u8, 10u8, 11u8, 20u8, 19u8];
        let inc = build_incremental(&net, &moves);
        let scr = build_from_scratch(&net, &moves);
        assert_eq!(inc.acc[0], scr.acc[0], "acc[0] mismatch at depth 6");
        assert_eq!(inc.acc[1], scr.acc[1], "acc[1] mismatch at depth 6");
    }

    /// Think should not corrupt the external acc array.
    #[test]
    fn think_does_not_corrupt_external_acc() {
        let net = load_net();
        let board = TicTacToe::new();
        let before = DualAccumulator::new(&net, &board);

        let mut search = Search::new();
        search.acc[0] = before;
        search.think(&board, 4, &net);

        // search.acc[0] must be unchanged
        assert_eq!(search.acc[0].acc[0], before.acc[0]);
        assert_eq!(search.acc[0].acc[1], before.acc[1]);
    }

    /// Verifies that the STM half returns a sensible (finite, in-range) score.
    #[test]
    fn forward_produces_valid_score() {
        let net = load_net();
        let board = TicTacToe::new();
        let acc = DualAccumulator::new(&net, &board);

        let bucket = get_bucket(board.ply);
        let score = net.forward(acc.stm(Symbol::Cross), bucket);
        assert!(score.is_finite(), "score must be finite");
        assert!((0.0..=1.0).contains(&score), "score must be in [0, 1]");
    }
}
