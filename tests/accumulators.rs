#[cfg(test)]
mod tests {
    use test_case::test_case;
    use ultimate_tic_tac_toe::{
        core::TicTacToe,
        network::{Accumulator, Network},
        search::Search,
    };

    #[test_case(Accumulator([0.0; 128]); "No one played anything yet")]
    fn accumulator_tests_init(expected: Accumulator) {
        let search = Search::new();
        assert_eq!(expected, search.acc[0]);
    }

    #[test_case(Accumulator([0.0; 128]))]
    fn accumulator_tests_play_one_move(mut expected: Accumulator) {
        let mut board = TicTacToe::new();
        let mut search = Search::new();
        let net = Network::load("databin/gen0_weights.bin".to_owned());

        let delta = board.make(0);
        search.acc[board.ply].apply_delta(&net, &delta);

        expected.add_features(&net, &[0, 189]);
        assert_eq!(expected, search.acc[1]);
    }

    #[test_case(Accumulator([0.0; 128]) ; "looking for accumulators corruption during think")]
    fn accumulator_tests_think_move(mut expected: Accumulator) {
        let mut board = TicTacToe::new();
        let mut search = Search::new();
        let net = Network::load("databin/gen0_weights.bin".to_owned());

        search.think(&board, 4, &net); // shouldnt impact the accumulators
        let delta = board.make(0);
        search.acc[board.ply].apply_delta(&net, &delta);

        expected.add_features(&net, &[0, 189]);
        assert_eq!(expected, search.acc[1]);
    }

    #[test_case(Accumulator([0.0; 128]) ; "looking for accumulators corruption during think")]
    fn accumulator_tests_play_think_move_think(mut expected: Accumulator) {
        let mut board = TicTacToe::new();
        let mut search = Search::new();
        let net = Network::load("databin/gen0_weights.bin".to_owned());

        search.think(&board, 4, &net); // shouldnt impact the accumulators
        let delta = board.make(0);
        search.acc[board.ply].apply_delta(&net, &delta);
        search.think(&board, 4, &net); // shouldnt impact the accumulators

        expected.add_features(&net, &[0, 189]);
        assert_eq!(expected, search.acc[1]);
    }

    #[test_case(Accumulator([0.0; 128]) ; "well init accumulators in search")]
    fn accumulator_tests_init_acc(expected: Accumulator) {
        let search = Search::new();
        assert_eq!(true, search.acc.iter().any(|&acc| acc == expected))
    }
}
