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
pub async fn generate_endpoint(axum::Json(request): axum::Json<GenerationRequest>) -> axum::Json<GenerationResponse> {
    let _ = (request.style, request.size);
    axum::Json(GenerationResponse {
        svg: r#"<svg viewBox="0 0 24 24"><path d="M4 12 L12 4 L20 12 L20 20 L4 20 Z"/></svg>"#.to_string(),
        valid: true,
        model: "SVGen 0.1".to_string(),
    })
}
