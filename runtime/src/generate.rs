use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

const REQUIRED_ARTIFACTS: &[&str] = &[
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "generation_config.json",
];
const FALLBACK_WARNING: &str = "Using fallback placeholder because no trained model is loaded.";

#[derive(Debug, PartialEq, Eq)]
pub enum ModelArtifactState {
    MissingArtifacts(Vec<String>),
    Placeholder,
    TrainedNoRuntimeInference,
}

#[derive(Debug, Deserialize)]
struct ExportConfig {
    status: Option<String>,
    runtime_inference: Option<String>,
}

pub fn fallback_home_svg() -> &'static str {
    r#"<svg viewBox="0 0 24 24"><path d="M4 12 L12 4 L20 12 L20 20 L4 20 Z"/></svg>"#
}

pub fn generate_svg(model_dir: &Path, prompt: &str) -> Result<String> {
    let _state = inspect_model_artifacts(model_dir)?;
    let _ = prompt;
    eprintln!("{FALLBACK_WARNING}");
    Ok(fallback_home_svg().to_string())
}

pub fn inspect_model_artifacts(model_dir: &Path) -> Result<ModelArtifactState> {
    let mut missing = Vec::new();
    for artifact in REQUIRED_ARTIFACTS {
        let path = model_dir.join(artifact);
        if !path.is_file() {
            missing.push(path.display().to_string());
        }
    }
    if !missing.is_empty() {
        return Ok(ModelArtifactState::MissingArtifacts(missing));
    }

    let config_path = model_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: ExportConfig = serde_json::from_str(&config_text)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    if config.status.as_deref() != Some("trained") {
        return Ok(ModelArtifactState::Placeholder);
    }
    let _runtime_inference = config.runtime_inference.as_deref().unwrap_or("unavailable");
    Ok(ModelArtifactState::TrainedNoRuntimeInference)
}

#[cfg(test)]
mod tests {
    use super::{fallback_home_svg, inspect_model_artifacts, ModelArtifactState};

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_artifacts(dir: &std::path::Path, config: &str) {
        std::fs::write(dir.join("model.safetensors"), b"model").unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.join("generation_config.json"), "{}").unwrap();
    }

    #[test]
    fn fallback_svg_is_valid_root_svg() {
        assert!(fallback_home_svg().starts_with(r#"<svg viewBox="0 0 24 24">"#));
    }

    #[test]
    fn detects_missing_artifacts() {
        let dir = temp_model_dir("svgen_missing_artifact_state");
        let state = inspect_model_artifacts(&dir).unwrap();
        assert!(matches!(state, ModelArtifactState::MissingArtifacts(_)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn detects_placeholder_artifacts() {
        let dir = temp_model_dir("svgen_placeholder_artifact_state");
        write_artifacts(&dir, r#"{"status":"placeholder"}"#);
        assert_eq!(
            inspect_model_artifacts(&dir).unwrap(),
            ModelArtifactState::Placeholder
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn detects_trained_artifacts_without_runtime_inference() {
        let dir = temp_model_dir("svgen_trained_artifact_state");
        write_artifacts(
            &dir,
            r#"{"status":"trained","runtime_inference":"unavailable"}"#,
        );
        assert_eq!(
            inspect_model_artifacts(&dir).unwrap(),
            ModelArtifactState::TrainedNoRuntimeInference
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
