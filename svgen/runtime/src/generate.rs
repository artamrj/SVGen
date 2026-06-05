use std::path::Path;

use anyhow::Result;

pub fn generate_svg(model_dir: &Path, prompt: &str) -> Result<String> {
    let _ = (model_dir, prompt);
    Ok(r#"<svg viewBox="0 0 24 24"><path d="M4 12 L12 4 L20 12 L20 20 L4 20 Z"/></svg>"#.to_string())
}
