use ultimate_tic_tac_toe::*;

fn main() -> anyhow::Result<()> {
    train::generate_databin()?;
    Ok(())
}
