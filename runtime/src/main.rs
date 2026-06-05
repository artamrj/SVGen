mod api;
mod generate;
mod postprocess;
mod tokenizer;
mod validator;

use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "svgen", version, about = "SVGen text-to-SVG runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate { model, prompt, out } => {
            let raw_svg = generate::generate_svg(&model, &prompt)?;
            let svg = postprocess::normalize_svg(&raw_svg)?;
            validator::validate_svg(&svg)?;
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
            }
            fs::write(&out, svg).with_context(|| format!("writing {}", out.display()))?;
        }
    }
    Ok(())
}
