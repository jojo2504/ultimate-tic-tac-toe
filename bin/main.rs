use ultimate_tic_tac_toe::{game::start_self_game_with_net, network::Network};

fn main() {
    let net = Network::load(format!("databin/gen{}_weights.bin", 0));
    for _ in 0..1 {
        start_self_game_with_net(&net, 6);
    }
}
