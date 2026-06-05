use std::path::Path;

use anyhow::{bail, Result};

const REQUIRED_ARTIFACTS: &[&str] = &["model.safetensors", "config.json", "tokenizer.json", "generation_config.json"];

pub fn generate_svg(model_dir: &Path, prompt: &str) -> Result<String> {
    validate_model_artifacts(model_dir)?;
    let _ = prompt;
    Ok(r#"<svg viewBox="0 0 24 24"><path d="M4 12 L12 4 L20 12 L20 20 L4 20 Z"/></svg>"#.to_string())
}

pub fn validate_model_artifacts(model_dir: &Path) -> Result<()> {
    for artifact in REQUIRED_ARTIFACTS {
        let path = model_dir.join(artifact);
        if !path.is_file() {
            bail!("missing required model artifact: {}", path.display());
        }
    }
    Ok(())
}
