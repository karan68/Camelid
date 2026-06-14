//! Parse a model's generated text into tool calls (Hybrid Phase 1). The server
//! renders tool definitions through the model's own chat template; this turns the
//! model's *output* back into structured calls. Family-specific: Llama 3.x emits
//! JSON (`{"name":…,"parameters":…}`, optionally `<|python_tag|>`-wrapped);
//! Qwen3/Hermes emit `<tool_call>{…}</tool_call>`. Malformed output yields no
//! calls (the loop then treats the text as a final answer) — never a panic.

use serde_json::Value;

use super::tools::ToolCall;

/// Parse `text` into zero or more tool calls. Empty = no tool call (plain answer).
pub fn parse(text: &str, family: &str) -> Vec<ToolCall> {
    let hermes_first = family.contains("qwen") || family.contains("hermes");
    if hermes_first {
        let calls = parse_hermes(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_json(text);
    }
    let calls = parse_json(text);
    if !calls.is_empty() {
        return calls;
    }
    parse_hermes(text)
}

/// `<tool_call>{ "name": …, "arguments": { … } }</tool_call>` blocks (Qwen/Hermes).
fn parse_hermes(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let inner = match rest.find("</tool_call>") {
            Some(end) => {
                let inner = &rest[..end];
                rest = &rest[end + "</tool_call>".len()..];
                inner
            }
            None => rest,
        };
        if let Ok(value) = serde_json::from_str::<Value>(inner.trim()) {
            if let Some(call) = call_from_obj(&value) {
                calls.push(call);
            }
        }
    }
    calls
}

/// Bare/`python_tag`-wrapped JSON tool call(s) (Llama 3.x).
fn parse_json(text: &str) -> Vec<ToolCall> {
    let cleaned = strip_markers(text);
    let trimmed = cleaned.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return calls_from_value(&value);
    }
    // Otherwise try to extract the first balanced {…} object.
    if let Some(slice) = first_json_object(trimmed) {
        if let Ok(value) = serde_json::from_str::<Value>(slice) {
            return calls_from_value(&value);
        }
    }
    Vec::new()
}

fn calls_from_value(value: &Value) -> Vec<ToolCall> {
    match value {
        Value::Array(items) => items.iter().filter_map(call_from_obj).collect(),
        Value::Object(_) => call_from_obj(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Build a call from an object: `name` + args from `parameters`/`arguments`/the
/// object minus the envelope keys. Returns None if there's no usable name.
fn call_from_obj(value: &Value) -> Option<ToolCall> {
    let obj = value.as_object()?;
    // Some models nest under "function": {"name":…,"arguments":…}.
    if let Some(func) = obj.get("function").and_then(Value::as_object) {
        let name = func.get("name").and_then(Value::as_str)?.to_string();
        let args = func
            .get("arguments")
            .or_else(|| func.get("parameters"))
            .cloned()
            .map(coerce_args)
            .unwrap_or_else(|| Value::Object(Default::default()));
        return Some(ToolCall { name, args });
    }
    let name = obj.get("name").and_then(Value::as_str)?.to_string();
    let args = obj
        .get("parameters")
        .or_else(|| obj.get("arguments"))
        .cloned()
        .map(coerce_args)
        .unwrap_or_else(|| {
            let mut rest = obj.clone();
            rest.remove("name");
            rest.remove("type");
            Value::Object(rest)
        });
    Some(ToolCall { name, args })
}

/// Arguments are sometimes a JSON *string* — decode it to an object when so.
fn coerce_args(value: Value) -> Value {
    if let Value::String(s) = &value {
        if let Ok(parsed) = serde_json::from_str::<Value>(s) {
            return parsed;
        }
    }
    value
}

fn strip_markers(text: &str) -> String {
    let mut s = text.to_string();
    for marker in [
        "<|python_tag|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "```json",
        "```",
    ] {
        s = s.replace(marker, " ");
    }
    s
}

/// First balanced `{…}` substring (depth-aware, ignores braces in strings).
fn first_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_json_with_parameters() {
        let out = parse(
            "<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/main.rs\"}}<|eom_id|>",
            "llama_bpe_decoder",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "src/main.rs");
    }

    #[test]
    fn parses_hermes_qwen_tool_call_tags() {
        let out = parse(
            "sure<tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}</tool_call>",
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], ".");
    }

    #[test]
    fn parses_call_embedded_in_prose() {
        let out = parse(
            "I will read it. {\"name\":\"read_file\",\"parameters\":{\"path\":\"a\"}} done",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn plain_answer_yields_no_calls() {
        assert!(parse("The file has 3 lines.", "llama").is_empty());
    }

    #[test]
    fn malformed_json_is_clean_not_a_panic() {
        // Looks like a call but is broken JSON → no calls, no panic.
        assert!(parse("{\"name\": \"read_file\", \"parameters\": {bad", "llama").is_empty());
        assert!(parse("<tool_call>{not json}</tool_call>", "qwen").is_empty());
    }
}
