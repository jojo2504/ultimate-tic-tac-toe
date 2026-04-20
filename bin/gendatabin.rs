use serde_json::Value;
use std::fs;
use ultimate_tic_tac_toe::*;

fn main() -> anyhow::Result<()> {
    let mut games_per_gen = 3000;
    if let Ok(config_str) = fs::read_to_string("config/config.json") {
        if let Ok(config) = serde_json::from_str::<Value>(&config_str) {
            if let Some(g) = config["depth"]["3"]["games_per_generation"].as_i64() {
                games_per_gen = g as i32;
            }
        }
    }

    train::generate_first_databin(0, games_per_gen)?;
    Ok(())
}
