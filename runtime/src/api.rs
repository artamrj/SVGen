#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub style: Option<String>,
    pub size: Option<u16>,
}

#[derive(Debug, Serialize)]
pub struct GenerationResponse {
    pub svg: String,
    pub valid: bool,
    pub model: String,
}

#[cfg(feature = "api")]
pub async fn generate_endpoint(
    axum::Json(request): axum::Json<GenerationRequest>,
) -> axum::Json<GenerationResponse> {
    let _ = (request.prompt, request.style, request.size);
    axum::Json(GenerationResponse {
        svg: crate::generate::fallback_home_svg().to_string(),
        valid: true,
        model: "SVGen 0.2.5 Tiny Generator".to_string(),
    })
}
