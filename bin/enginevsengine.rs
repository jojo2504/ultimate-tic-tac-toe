use ultimate_tic_tac_toe::train::tournament;

fn main() {
    let engine1 = String::from("databin/gen0_weights.bin");
    let engine2 = String::from("databin/gen0_weights.bin");
    tournament(&engine1, &engine2, 15);
}
