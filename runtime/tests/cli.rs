use std::process::Command;

#[test]
fn cli_parses_generate_arguments() {
    let model_dir = std::env::temp_dir().join("svgen_cli_model");
    std::fs::create_dir_all(&model_dir).unwrap();
    for name in ["model.safetensors", "config.json", "tokenizer.json", "generation_config.json"] {
        std::fs::write(model_dir.join(name), "{}").unwrap();
    }
    let temp = std::env::temp_dir().join("svgen_cli_test.svg");
    let status = Command::new(env!("CARGO_BIN_EXE_svgen"))
        .args([
            "generate",
            "--model",
            model_dir.to_str().unwrap(),
            "--prompt",
            "minimal rocket icon",
            "--out",
            temp.to_str().unwrap(),
        ])
        .status()
        .expect("run svgen");
    assert!(status.success());
    let _ = std::fs::remove_file(temp);
    let _ = std::fs::remove_dir_all(model_dir);
}

#[test]
fn cli_fails_when_artifacts_are_missing() {
    let model_dir = std::env::temp_dir().join("svgen_missing_model");
    let _ = std::fs::remove_dir_all(&model_dir);
    std::fs::create_dir_all(&model_dir).unwrap();
    let temp = std::env::temp_dir().join("svgen_missing_test.svg");
    let status = Command::new(env!("CARGO_BIN_EXE_svgen"))
        .args([
            "generate",
            "--model",
            model_dir.to_str().unwrap(),
            "--prompt",
            "minimal rocket icon",
            "--out",
            temp.to_str().unwrap(),
        ])
        .status()
        .expect("run svgen");
    assert!(!status.success());
    let _ = std::fs::remove_dir_all(model_dir);
}
