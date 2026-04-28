// Focused gauntlet evaluation for a single challenger generation.
//
// Use this when you want a fast pass/fail signal — e.g. inside the auto-train
// loop, or after manually training one new generation.
//
// Plays the challenger against a fixed-shape panel:
//   - Gen 0 (heuristic baseline — "have we forgotten how to play vs random/heuristic?")
//   - N-20 (long memory)
//   - N-10 (medium memory)
//   - N-5  (recent memory)
//   - N-1  (most recent — "are we improving on the previous gen?")
//   - "best" (the current promoted champion)
//
// For each opponent, runs `games_per_opponent` games (split equally as Cross/Circle)
// and prints W/D/L + Elo diff. Aggregate verdict:
//   PASS  if avg Elo > +PASS_THRESHOLD AND no opponent below -CATASTROPHE_THRESHOLD
//   FAIL  otherwise
//
// Usage:
//   cargo run --release --bin gauntlet -- <challenger_gen> [best_gen] [games_per_opponent]
//   cargo run --release --bin gauntlet -- 42 41 200

use colored::Colorize;
use std::{env, fs, process::ExitCode};
use ultimate_tic_tac_toe::train::tournament;

/// Aggregate Elo above which the challenger passes.
const PASS_THRESHOLD: f32 = 0.0;
/// If any opponent inflicts more than this Elo deficit, fail unconditionally
/// (catches regressions that are masked by easy wins elsewhere).
const CATASTROPHE_THRESHOLD: f32 = 80.0;
/// Default games per opponent if not specified on the CLI.
const DEFAULT_GAMES: u32 = 200;
/// Search depth used for all gauntlet games.
const DEPTH: i32 = 3;

fn weights_path(gen: i32) -> String {
    format!("databin/gen{gen}_weights.bin")
}

fn weights_exist(gen: i32) -> bool {
    fs::metadata(weights_path(gen)).is_ok()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <challenger_gen> [best_gen] [games_per_opponent]",
            args[0]
        );
        return ExitCode::from(2);
    }

    let challenger: i32 = match args[1].parse() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("invalid challenger gen: {}", args[1]);
            return ExitCode::from(2);
        }
    };

    // Best gen defaults to challenger - 1 if not provided
    let best: i32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(challenger - 1);

    let games_per_opponent: u32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_GAMES);

    // ── Build opponent panel ──────────────────────────────────────────────
    let mut panel: Vec<(&'static str, i32)> = vec![];
    if challenger >= 1 && weights_exist(0) {
        panel.push(("gen0  (baseline)", 0));
    }
    if challenger >= 21 && weights_exist(challenger - 20) {
        panel.push(("N-20  (long mem)", challenger - 20));
    }
    if challenger >= 11 && weights_exist(challenger - 10) {
        panel.push(("N-10  (med mem)", challenger - 10));
    }
    if challenger >= 6 && weights_exist(challenger - 5) {
        panel.push(("N-5   (recent)", challenger - 5));
    }
    if challenger >= 2 && weights_exist(challenger - 1) {
        panel.push(("N-1   (previous)", challenger - 1));
    }
    if best != challenger - 1 && weights_exist(best) {
        panel.push(("best  (champion)", best));
    }

    if panel.is_empty() {
        eprintln!("no eligible opponents found — is gen{challenger} the very first gen?");
        return ExitCode::from(2);
    }

    if !weights_exist(challenger) {
        eprintln!("challenger weights not found: {}", weights_path(challenger));
        return ExitCode::from(2);
    }

    println!(
        "{}",
        format!(
            "Gauntlet: gen{challenger} vs {} opponents, {} games each, depth {}",
            panel.len(),
            games_per_opponent,
            DEPTH
        )
        .bold()
    );
    println!("{}", "─".repeat(60));

    // ── Run tournaments ───────────────────────────────────────────────────
    let mut results: Vec<(&'static str, i32, f32)> = vec![];
    let mut min_elo = f32::INFINITY;
    let mut sum_elo = 0.0f32;

    let challenger_path = weights_path(challenger);
    for (label, opp_gen) in &panel {
        let opp_path = weights_path(*opp_gen);
        print!("{:18}: ", label);
        // tournament() prints its own W/D/L line
        let elo = tournament(&opp_path, &challenger_path, games_per_opponent, DEPTH);
        results.push((label, *opp_gen, elo));
        sum_elo += elo;
        if elo < min_elo {
            min_elo = elo;
        }
    }

    let avg_elo = sum_elo / results.len() as f32;

    // ── Verdict ──────────────────────────────────────────────────────────
    println!("{}", "─".repeat(60));
    println!("{}", "Per-opponent breakdown:".bold());
    for (label, opp_gen, elo) in &results {
        let line = format!("  {:18}  (gen{:>3}): {:+7.1} Elo", label, opp_gen, elo);
        if *elo >= 50.0 {
            println!("{}", line.green());
        } else if *elo >= -CATASTROPHE_THRESHOLD {
            println!("{}", line.yellow());
        } else {
            println!("{}", line.red());
        }
    }

    println!("{}", "─".repeat(60));
    println!(
        "Aggregate: avg {:+.1} Elo  |  worst {:+.1} Elo",
        avg_elo, min_elo
    );

    let pass = avg_elo > PASS_THRESHOLD && min_elo > -CATASTROPHE_THRESHOLD;
    if pass {
        println!(
            "{}",
            format!("VERDICT: PASS — promote gen{challenger}")
                .green()
                .bold()
        );
        ExitCode::SUCCESS
    } else {
        let reason = if min_elo <= -CATASTROPHE_THRESHOLD {
            format!(
                "catastrophic regression vs at least one opponent ({:+.1} Elo ≤ -{:.0})",
                min_elo, CATASTROPHE_THRESHOLD
            )
        } else {
            format!(
                "aggregate Elo {:+.1} below threshold {:+.1}",
                avg_elo, PASS_THRESHOLD
            )
        };
        println!(
            "{}",
            format!("VERDICT: FAIL — reject gen{challenger}\n  reason: {reason}")
                .red()
                .bold()
        );
        ExitCode::from(1)
    }
}
