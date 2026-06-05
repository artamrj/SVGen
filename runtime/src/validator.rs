use std::str;

use anyhow::{anyhow, bail, Context, Result};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;

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
    let mut reader = Reader::from_str(svg);
    reader.config_mut().trim_text(true);
    let mut root_seen = false;
    let mut depth = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => {
                validate_element(&element)?;
                let tag = element_name(&element)?;
                if depth == 0 {
                    if tag != "svg" {
                        bail!("root element must be svg");
                    }
                    root_seen = true;
                }
                depth += 1;
            }
            Ok(Event::Empty(element)) => {
                validate_element(&element)?;
                let tag = element_name(&element)?;
                if depth == 0 {
                    if tag != "svg" {
                        bail!("root element must be svg");
                    }
                    root_seen = true;
                }
            }
            Ok(Event::End(_)) => {
                depth = depth.saturating_sub(1);
            }
            Ok(Event::Eof) => break,
            Ok(Event::Text(_)) | Ok(Event::Comment(_)) | Ok(Event::Decl(_)) => {}
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
    if !ALLOWED_TAGS.contains(&tag) {
        bail!("unsupported SVG tag: {tag}");
    }

    for attr in element.attributes() {
        let attr = attr.context("invalid XML attribute")?;
        let key = str::from_utf8(attr.key.as_ref()).context("attribute name is not UTF-8")?;
        if !ALLOWED_ATTRS.contains(&key) {
            bail!("unsupported SVG attribute: {key}");
        }
        let value = str::from_utf8(attr.value.as_ref()).context("attribute value is not UTF-8")?;
        validate_attribute_value(key, value)?;
    }
    Ok(())
}

fn element_name(element: &BytesStart<'_>) -> Result<&str> {
    str::from_utf8(element.name().as_ref()).context("element name is not UTF-8")
}

fn validate_attribute_value(key: &str, value: &str) -> Result<()> {
    let lower = value.to_ascii_lowercase();
    if lower.contains("javascript:") || lower.contains("data:") || lower.contains("http://") || lower.contains("https://") {
        bail!("unsafe external reference in attribute: {key}");
    }
    if key == "viewBox" && value.trim() != "0 0 24 24" {
        bail!("viewBox must be 0 0 24 24");
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
}
