use std::str;

use anyhow::{anyhow, bail, Context, Result};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

const MAX_SVG_BYTES: usize = 16 * 1024;
const MAX_DEPTH: usize = 8;
const MAX_CHILDREN: usize = 128;
const MAX_PATH_D_LEN: usize = 2048;
const MAX_NUMERIC_ABS: f64 = 10000.0;

const ALLOWED_TAGS: &[&str] = &["svg", "path", "circle", "rect", "line", "polyline", "polygon", "g"];
const ALLOWED_ATTRS: &[&str] = &[
    "viewBox",
    "d",
    "cx",
    "cy",
    "r",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "points",
    "fill",
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "transform",
    "opacity",
];

pub fn validate_svg(svg: &str) -> Result<()> {
    if svg.as_bytes().len() > MAX_SVG_BYTES {
        bail!("SVG exceeds maximum size");
    }
    let mut reader = Reader::from_str(svg);
    reader.config_mut().trim_text(true);
    let mut root_seen = false;
    let mut depth = 0usize;
    let mut children = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => {
                validate_element(&element)?;
                let tag = element_name(&element)?;
                if depth == 0 {
                    if tag != "svg" {
                        bail!("root element must be svg");
                    }
                    if root_seen {
                        bail!("multiple svg roots are not allowed");
                    }
                    root_seen = true;
                }
                depth += 1;
                children += 1;
                if depth > MAX_DEPTH {
                    bail!("SVG exceeds maximum depth");
                }
                if children > MAX_CHILDREN {
                    bail!("SVG exceeds maximum child count");
                }
            }
            Ok(Event::Empty(element)) => {
                validate_element(&element)?;
                let tag = element_name(&element)?;
                if depth == 0 {
                    if tag != "svg" {
                        bail!("root element must be svg");
                    }
                    if root_seen {
                        bail!("multiple svg roots are not allowed");
                    }
                    root_seen = true;
                }
                children += 1;
                if children > MAX_CHILDREN {
                    bail!("SVG exceeds maximum child count");
                }
            }
            Ok(Event::End(_)) => {
                depth = depth.saturating_sub(1);
            }
            Ok(Event::Eof) => break,
            Ok(Event::Text(text)) => {
                if !text.as_ref().iter().all(u8::is_ascii_whitespace) {
                    bail!("text nodes are not allowed");
                }
            }
            Ok(Event::Comment(_)) | Ok(Event::Decl(_)) => {}
            Ok(_) => bail!("unsupported XML event"),
            Err(error) => return Err(anyhow!("invalid XML: {error}")),
        }
    }

    if !root_seen {
        bail!("missing svg root");
    }
    Ok(())
}

fn validate_element(element: &BytesStart<'_>) -> Result<()> {
    let tag = element_name(element)?;
    if tag.contains(':') {
        bail!("namespace-prefixed SVG tags are not allowed: {tag}");
    }
    if !ALLOWED_TAGS.contains(&tag.as_str()) {
        bail!("unsupported SVG tag: {tag}");
    }

    for attr in element.attributes() {
        let attr = attr.context("invalid XML attribute")?;
        let key = str::from_utf8(attr.key.as_ref()).context("attribute name is not UTF-8")?;
        if key.contains(':') || key.to_ascii_lowercase().starts_with("on") {
            bail!("unsafe SVG attribute: {key}");
        }
        if !ALLOWED_ATTRS.contains(&key) {
            bail!("unsupported SVG attribute: {key}");
        }
        let value = str::from_utf8(attr.value.as_ref()).context("attribute value is not UTF-8")?;
        validate_attribute_value(key, value)?;
    }
    Ok(())
}

fn element_name(element: &BytesStart<'_>) -> Result<String> {
    Ok(str::from_utf8(element.name().as_ref())
        .context("element name is not UTF-8")?
        .to_string())
}

fn validate_attribute_value(key: &str, value: &str) -> Result<()> {
    let lower = value.to_ascii_lowercase();
    if lower.contains("javascript:") || lower.contains("data:") || lower.contains("http://") || lower.contains("https://") {
        bail!("unsafe external reference in attribute: {key}");
    }
    if key == "d" && value.len() > MAX_PATH_D_LEN {
        bail!("path data exceeds maximum length");
    }
    if key == "viewBox" && value.trim() != "0 0 24 24" {
        bail!("viewBox must be 0 0 24 24");
    }
    validate_numeric_limits(value)?;
    Ok(())
}

fn validate_numeric_limits(value: &str) -> Result<()> {
    for token in value.split(|character: char| !(character.is_ascii_digit() || character == '.' || character == '-')) {
        if token.is_empty() || token == "-" || token == "." {
            continue;
        }
        if let Ok(number) = token.parse::<f64>() {
            if number.abs() > MAX_NUMERIC_ABS {
                bail!("numeric attribute value exceeds maximum absolute value");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_svg;

    #[test]
    fn accepts_safe_svg() {
        let svg = r#"<svg viewBox="0 0 24 24"><path d="M3 12 L12 3 Z" fill="none" stroke="currentColor"/></svg>"#;
        assert!(validate_svg(svg).is_ok());
    }

    #[test]
    fn rejects_script() {
        let svg = r#"<svg viewBox="0 0 24 24"><script>alert(1)</script></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_foreign_object() {
        let svg = r#"<svg viewBox="0 0 24 24"><foreignObject /></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_external_links() {
        let svg = r#"<svg viewBox="0 0 24 24"><path href="https://example.com/x" d="M0 0"/></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_unsupported_features() {
        let svg = r#"<svg viewBox="0 0 24 24"><filter id="shadow"/></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_namespace_prefixed_tag() {
        let svg = r#"<svg viewBox="0 0 24 24"><x:script /></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_event_attribute() {
        let svg = r#"<svg viewBox="0 0 24 24" onload="alert(1)"><path d="M0 0"/></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_excessive_numeric_value() {
        let svg = r#"<svg viewBox="0 0 24 24"><circle cx="10001" cy="12" r="2"/></svg>"#;
        assert!(validate_svg(svg).is_err());
    }

    #[test]
    fn rejects_excessive_path_length() {
        let d = "M0 0 ".repeat(500);
        let svg = format!(r#"<svg viewBox="0 0 24 24"><path d="{d}"/></svg>"#);
        assert!(validate_svg(&svg).is_err());
    }
}
