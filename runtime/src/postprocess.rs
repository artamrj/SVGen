use anyhow::{bail, Result};

use crate::validator::validate_svg;

const WRAPPABLE_TAGS: &[&str] = &["path", "circle", "rect", "line", "polyline", "polygon", "g"];

pub fn normalize_svg(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if validate_svg(trimmed).is_ok() {
        return Ok(trimmed.to_string());
    }

    if is_wrappable_safe_element(trimmed) {
        let wrapped = format!(r#"<svg viewBox="0 0 24 24">{trimmed}</svg>"#);
        validate_svg(&wrapped)?;
        return Ok(wrapped);
    }

    bail!("SVG output could not be normalized safely")
}

fn is_wrappable_safe_element(raw: &str) -> bool {
    if raw.contains(':') || raw.contains("<svg") || raw.contains("<?xml") {
        return false;
    }
    WRAPPABLE_TAGS.iter().any(|tag| {
        raw.starts_with(&format!("<{tag}"))
            && (raw.ends_with("/>") || raw.ends_with(&format!("</{tag}>")))
    })
}

#[cfg(test)]
mod tests {
    use super::normalize_svg;

    #[test]
    fn wraps_bare_safe_path() {
        let svg = normalize_svg(r#"<path d="M3 12 L12 3 Z"/>"#).unwrap();
        assert!(svg.starts_with(r#"<svg viewBox="0 0 24 24">"#));
    }

    #[test]
    fn refuses_unsafe_bare_element() {
        assert!(normalize_svg(r#"<script>alert(1)</script>"#).is_err());
    }

    #[test]
    fn refuses_namespace_prefixed_fragment() {
        assert!(normalize_svg(r#"<x:path d="M0 0"/>"#).is_err());
    }
}
