# UTTT - Ultimate Tic Tac Toe

## Getting Started

To get any net, do as follow:

```sh
cargo run --bin bootstrap --release 
uv venv
uv pip install -r requirements.txt
python train.py 0
cargo run --bin training_loop --release
```

## Build from source

```sh
cargo build --release
./target/release/enginevsplayer {gen} {depth}
```

or

```sh
cargo run --bin enginevsplayer {gen} {depth} --release
```
