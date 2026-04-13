use ultimate_tic_tac_toe::{
    core::TicTacToe, game::start_self_game_with_net, network::Network, search::Search,
};

fn main() {
    let net = Network::load(format!("databin/gen{}_weights.bin", 0));
    for _ in 0..1 {
        start_self_game_with_net(&net);
    }
}
