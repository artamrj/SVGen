use std::process::Command;

const FALLBACK_WARNING: &str = "Using fallback placeholder because no trained model is loaded.";

#[test]
fn cli_parses_generate_arguments() {
    let model_dir = std::env::temp_dir().join("svgen_cli_model");
    std::fs::create_dir_all(&model_dir).unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "generation_config.json",
    ] {
        std::fs::write(model_dir.join(name), "{}").unwrap();
    }
    let temp = std::env::temp_dir().join("svgen_cli_test.svg");
    let output = Command::new(env!("CARGO_BIN_EXE_svgen"))
        .args([
            "generate",
            "--model",
            model_dir.to_str().unwrap(),
            "--prompt",
            "minimal rocket icon",
            "--out",
            temp.to_str().unwrap(),
        ])
        .output()
        .expect("run svgen");
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains(FALLBACK_WARNING));
    let svg = std::fs::read_to_string(&temp).unwrap();
    assert!(svg.contains(r#"<svg viewBox="0 0 24 24">"#));
    let _ = std::fs::remove_file(temp);
    let _ = std::fs::remove_dir_all(model_dir);
}

#[test]
fn cli_uses_fallback_when_artifacts_are_missing() {
    let model_dir = std::env::temp_dir().join("svgen_missing_model");
    let _ = std::fs::remove_dir_all(&model_dir);
    std::fs::create_dir_all(&model_dir).unwrap();
    let temp = std::env::temp_dir().join("svgen_missing_test.svg");
    let output = Command::new(env!("CARGO_BIN_EXE_svgen"))
        .args([
            "generate",
            "--model",
            model_dir.to_str().unwrap(),
            "--prompt",
            "minimal rocket icon",
            "--out",
            temp.to_str().unwrap(),
        ])
        .output()
        .expect("run svgen");
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains(FALLBACK_WARNING));
    let svg = std::fs::read_to_string(&temp).unwrap();
    assert!(svg.contains(r#"<svg viewBox="0 0 24 24">"#));
    let _ = std::fs::remove_file(temp);
    let _ = std::fs::remove_dir_all(model_dir);
}
