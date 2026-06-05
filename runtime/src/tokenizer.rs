#![allow(dead_code)]

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    artifact_path: String,
}

impl Tokenizer {
    pub fn from_file(path: impl Into<String>) -> Result<Self> {
        Ok(Self {
            artifact_path: path.into(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let _ = &self.artifact_path;
        Ok(text.bytes().map(u32::from).collect())
    }
}
