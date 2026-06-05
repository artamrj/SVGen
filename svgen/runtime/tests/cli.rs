use std::process::Command;

#[test]
fn cli_parses_generate_arguments() {
    let temp = std::env::temp_dir().join("svgen_cli_test.svg");
    let status = Command::new(env!("CARGO_BIN_EXE_svgen"))
        .args([
            "generate",
            "--model",
            "../exports/svgen-0.1",
            "--prompt",
            "minimal rocket icon",
            "--out",
            temp.to_str().unwrap(),
        ])
        .status()
        .expect("run svgen");
    assert!(status.success());
    let _ = std::fs::remove_file(temp);
}
