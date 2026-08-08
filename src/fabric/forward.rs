//! Forwarding a placed request to the node that will serve it.
//!
//! Placement ([`super::policy`]) chooses a node; this sends the request there and
//! brings the answer back. The whole point of the fabric is that this happens
//! once per request rather than once per token, so a node's own generation loop
//! never waits on the network.
//!
//! Streaming is deliberately out of scope for this path. An SSE response is a
//! different response shape and a different cancellation story; rather than
//! half-support it, a streaming request is refused with a reason.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::http::{self, HttpError};
use super::node::NodeSpec;

/// Generation can legitimately take minutes, so this is far longer than a probe
/// budget. It exists to bound a wedged node, not to bound a slow model.
pub const DEFAULT_FORWARD_TIMEOUT: Duration = Duration::from_secs(300);

/// A completion body is far larger than a health body but still bounded.
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// The node's answer, tagged with which node produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forwarded {
    pub label: String,
    /// The node's HTTP status. A node answering 503 is a real answer, not a
    /// forwarding failure, so it arrives here rather than as an error.
    pub status: u16,
    pub body: Value,
    pub elapsed: Duration,
}

impl Forwarded {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardError {
    /// The request never completed against that node.
    Transport { label: String, detail: String },
    /// The node answered, but not with JSON.
    Json { label: String, detail: String },
    /// Refused before sending; see [`reject_streaming`].
    Unsupported(String),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport { label, detail } => {
                write!(f, "node `{label}` did not answer: {detail}")
            }
            Self::Json { label, detail } => {
                write!(f, "node `{label}` returned an unreadable body: {detail}")
            }
            Self::Unsupported(detail) => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for ForwardError {}

/// Refuse a streaming request rather than mis-parse an SSE body as JSON. Pure.
pub fn reject_streaming(body: &Value) -> Result<(), ForwardError> {
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(ForwardError::Unsupported(
            "the fabric does not forward streaming requests yet; \
             send the request without `stream: true`"
                .to_string(),
        ));
    }
    Ok(())
}

/// Build a minimal OpenAI-shaped chat request. Pure.
pub fn chat_request(model: &str, prompt: &str, max_tokens: u32) -> Value {
    json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "stream": false,
    })
}

/// Pull the assistant text out of a chat completion body. Pure.
///
/// Returns `None` rather than a placeholder so a caller can tell "the node
/// answered with no content" from "the node said nothing useful".
pub fn completion_text(body: &Value) -> Option<&str> {
    body.get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
}

/// Read an engine error message out of a non-2xx body, if it carries one. Pure.
pub fn error_message(body: &Value) -> Option<&str> {
    body.get("error")
        .and_then(|error| error.get("message").or(Some(error)))
        .and_then(Value::as_str)
}

/// Send one request to one node.
///
/// `bearer` is required by any node started with an API key: `/v1/health` is
/// exempt from the server's auth but `/v1/chat/completions` is not, so without
/// it this is the call that comes back 401.
pub fn forward(
    spec: &NodeSpec,
    path: &str,
    body: &Value,
    bearer: Option<&str>,
    timeout: Duration,
) -> Result<Forwarded, ForwardError> {
    reject_streaming(body)?;

    let encoded = serde_json::to_vec(body).map_err(|error| ForwardError::Json {
        label: spec.label.clone(),
        detail: format!("request body could not be encoded: {error}"),
    })?;

    let started = Instant::now();
    let response = http::request(
        &spec.host,
        spec.port,
        "POST",
        path,
        Some(&encoded),
        bearer,
        timeout,
        MAX_RESPONSE_BYTES,
    )
    .map_err(|error: HttpError| ForwardError::Transport {
        label: spec.label.clone(),
        detail: error.to_string(),
    })?;
    let elapsed = started.elapsed();

    // An empty body is legal for some statuses; represent it as null rather than
    // failing, so the status still reaches the caller.
    let parsed = if response.body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice::<Value>(&response.body).map_err(|error| ForwardError::Json {
            label: spec.label.clone(),
            detail: error.to_string(),
        })?
    };

    Ok(Forwarded {
        label: spec.label.clone(),
        status: response.status,
        body: parsed,
        elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(label: &str, port: u16) -> NodeSpec {
        NodeSpec {
            label: label.to_string(),
            host: "127.0.0.1".to_string(),
            port,
        }
    }

    #[test]
    fn a_chat_request_carries_the_model_prompt_and_no_streaming() {
        let request = chat_request("llama-3b", "hello", 16);
        assert_eq!(request["model"], "llama-3b");
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "hello");
        assert_eq!(request["max_tokens"], 16);
        assert_eq!(request["stream"], false);
    }

    #[test]
    fn a_streaming_request_is_refused_before_a_socket_is_opened() {
        let body = json!({ "model": "m", "stream": true });
        let error = reject_streaming(&body).expect_err("streaming is unsupported");
        assert!(matches!(error, ForwardError::Unsupported(_)));

        // Port 1 is closed; a transport error here would prove we tried to send.
        let error = forward(
            &spec("a", 1),
            "/v1/chat/completions",
            &body,
            None,
            Duration::from_millis(200),
        )
        .expect_err("refused");
        assert!(
            matches!(error, ForwardError::Unsupported(_)),
            "must refuse before dialling, got {error:?}"
        );
    }

    #[test]
    fn a_non_streaming_request_is_allowed_through() {
        assert!(reject_streaming(&json!({ "model": "m" })).is_ok());
        assert!(reject_streaming(&json!({ "model": "m", "stream": false })).is_ok());
    }

    #[test]
    fn completion_text_reads_the_first_choice() {
        let body = json!({
            "choices": [{ "message": { "role": "assistant", "content": "hi there" } }]
        });
        assert_eq!(completion_text(&body), Some("hi there"));
    }

    #[test]
    fn completion_text_is_none_rather_than_a_placeholder_when_absent() {
        assert_eq!(completion_text(&json!({})), None);
        assert_eq!(completion_text(&json!({ "choices": [] })), None);
        assert_eq!(completion_text(&json!({ "choices": [{}] })), None);
    }

    #[test]
    fn an_engine_error_message_is_extracted_for_reporting() {
        let body = json!({ "error": { "message": "engine queue full" } });
        assert_eq!(error_message(&body), Some("engine queue full"));

        let flat = json!({ "error": "bad request" });
        assert_eq!(error_message(&flat), Some("bad request"));
    }

    #[test]
    fn a_dead_node_reports_which_node_failed() {
        let error = forward(
            &spec("windows", 1),
            "/v1/chat/completions",
            &chat_request("m", "hi", 4),
            None,
            Duration::from_millis(400),
        )
        .expect_err("port 1 is closed");
        match &error {
            ForwardError::Transport { label, .. } => assert_eq!(label, "windows"),
            other => panic!("expected Transport, got {other:?}"),
        }
        // The message must name the node, or an operator cannot act on it.
        assert!(error.to_string().contains("windows"), "{error}");
    }

    #[test]
    fn a_success_status_is_distinguished_from_an_engine_refusal() {
        let ok = Forwarded {
            label: "a".to_string(),
            status: 200,
            body: Value::Null,
            elapsed: Duration::ZERO,
        };
        let refused = Forwarded {
            status: 503,
            ..ok.clone()
        };
        assert!(ok.is_success());
        assert!(!refused.is_success());
    }
}
