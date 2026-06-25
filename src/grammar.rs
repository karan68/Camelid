//! Minimal JSON-object grammar for constrained decoding (`response_format:
//! {"type":"json_object"}`). A byte-level pushdown automaton that decides, at each
//! decode step, which token continuations keep the output a valid JSON-object
//! prefix — so the model can only emit well-formed JSON. Pure and heavily tested;
//! this is the high-value core of the structured-output lane. Arbitrary GBNF and
//! full JSON Schema are deliberate follow-ups.

/// The container we are currently inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Object,
    Array,
}

/// Sub-state while scanning a JSON number. The "complete" sub-states (a number may
/// validly end here) are `Zero`, `Int`, `Frac`, `Exp`; the rest require more input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Num {
    Sign,     // saw '-', need first digit
    Zero,     // saw leading '0' (complete)
    Int,      // saw 1-9 then digits (complete)
    DotFirst, // saw '.', need first fraction digit
    Frac,     // fraction digits (complete)
    ExpSign,  // saw e/E, may take +/- or a digit
    ExpFirst, // saw e/E and a sign, need first exponent digit
    Exp,      // exponent digits (complete)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Start,                                    // top level: whitespace then '{'
    Value,                                    // a value is required (after ':')
    Str { escape: bool, hex: u8, key: bool }, // inside a "string"
    Num(Num),                                 // inside a number
    Lit { rest: &'static [u8] },              // matching true / false / null
    Key { allow_close: bool },                // object: a key string (or '}')
    Colon,                                    // object: ':'
    ArrVal { allow_close: bool },             // array: a value (or ']')
    AfterValue,                               // ',' or the container's close
    Done,                                     // top-level object complete
}

/// Incremental JSON-object validator. `advance` consumes one byte and returns an
/// error if that byte cannot extend a valid JSON object. `is_done` is true once the
/// top-level object has fully closed.
#[derive(Clone, Debug)]
pub struct JsonState {
    stack: Vec<Frame>,
    mode: Mode,
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

fn is_hex(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// A byte that can terminate a number (whitespace or a structural close/separator).
fn is_num_delim(b: u8) -> bool {
    is_ws(b) || matches!(b, b',' | b'}' | b']')
}

impl JsonState {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            mode: Mode::Start,
        }
    }

    /// True when a complete top-level JSON object has been emitted (the model may
    /// stop). The decode loop stops as soon as this becomes true.
    pub fn is_done(&self) -> bool {
        self.mode == Mode::Done
    }

    /// Would `bytes` keep the output a valid JSON-object prefix from the current
    /// state? Used to mask a candidate token without mutating the live state.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        for &b in bytes {
            if probe.advance(b).is_err() {
                return false;
            }
        }
        true
    }

    /// Begin a value on `b` (used after ':' and after array '[' / ','). Strings
    /// started here are values (not keys).
    fn start_value(&mut self, b: u8) -> Result<(), ()> {
        self.mode = match b {
            b'{' => {
                self.stack.push(Frame::Object);
                Mode::Key { allow_close: true }
            }
            b'[' => {
                self.stack.push(Frame::Array);
                Mode::ArrVal { allow_close: true }
            }
            b'"' => Mode::Str {
                escape: false,
                hex: 0,
                key: false,
            },
            b'-' => Mode::Num(Num::Sign),
            b'0' => Mode::Num(Num::Zero),
            b'1'..=b'9' => Mode::Num(Num::Int),
            b't' => Mode::Lit { rest: b"rue" },
            b'f' => Mode::Lit { rest: b"alse" },
            b'n' => Mode::Lit { rest: b"ull" },
            _ => return Err(()),
        };
        Ok(())
    }

    /// A container (or the top-level object) just closed, completing a value.
    fn after_close(&mut self) {
        self.mode = if self.stack.is_empty() {
            Mode::Done
        } else {
            Mode::AfterValue
        };
    }

    /// Consume one byte; `Err(())` means the byte cannot extend a valid JSON object.
    /// The error is intentionally information-free — a rejected byte is a rejected
    /// byte; the caller only needs the yes/no.
    #[allow(clippy::result_unit_err)]
    pub fn advance(&mut self, b: u8) -> Result<(), ()> {
        match self.mode {
            Mode::Start => {
                if is_ws(b) {
                    Ok(())
                } else if b == b'{' {
                    self.stack.push(Frame::Object);
                    self.mode = Mode::Key { allow_close: true };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Mode::Value => {
                if is_ws(b) {
                    Ok(())
                } else {
                    self.start_value(b)
                }
            }
            Mode::ArrVal { allow_close } => {
                if is_ws(b) {
                    Ok(())
                } else if allow_close && b == b']' {
                    self.stack.pop();
                    self.after_close();
                    Ok(())
                } else {
                    self.start_value(b)
                }
            }
            Mode::Key { allow_close } => {
                if is_ws(b) {
                    Ok(())
                } else if allow_close && b == b'}' {
                    self.stack.pop();
                    self.after_close();
                    Ok(())
                } else if b == b'"' {
                    self.mode = Mode::Str {
                        escape: false,
                        hex: 0,
                        key: true,
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Mode::Colon => {
                if is_ws(b) {
                    Ok(())
                } else if b == b':' {
                    self.mode = Mode::Value;
                    Ok(())
                } else {
                    Err(())
                }
            }
            Mode::Str { escape, hex, key } => {
                if hex > 0 {
                    if is_hex(b) {
                        self.mode = Mode::Str {
                            escape: false,
                            hex: hex - 1,
                            key,
                        };
                        Ok(())
                    } else {
                        Err(())
                    }
                } else if escape {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.mode = Mode::Str {
                                escape: false,
                                hex: 0,
                                key,
                            };
                            Ok(())
                        }
                        b'u' => {
                            self.mode = Mode::Str {
                                escape: false,
                                hex: 4,
                                key,
                            };
                            Ok(())
                        }
                        _ => Err(()),
                    }
                } else {
                    match b {
                        b'"' => {
                            self.mode = if key { Mode::Colon } else { Mode::AfterValue };
                            Ok(())
                        }
                        b'\\' => {
                            self.mode = Mode::Str {
                                escape: true,
                                hex: 0,
                                key,
                            };
                            Ok(())
                        }
                        // Raw control characters are not allowed in a JSON string.
                        0x00..=0x1F => Err(()),
                        _ => Ok(()),
                    }
                }
            }
            Mode::Lit { rest } => match rest.split_first() {
                Some((&first, tail)) if first == b => {
                    if tail.is_empty() {
                        self.mode = Mode::AfterValue;
                    } else {
                        self.mode = Mode::Lit { rest: tail };
                    }
                    Ok(())
                }
                _ => Err(()),
            },
            Mode::Num(num) => self.advance_num(num, b),
            Mode::AfterValue => {
                if is_ws(b) {
                    return Ok(());
                }
                match self.stack.last().copied() {
                    Some(Frame::Object) => match b {
                        b',' => {
                            self.mode = Mode::Key { allow_close: false };
                            Ok(())
                        }
                        b'}' => {
                            self.stack.pop();
                            self.after_close();
                            Ok(())
                        }
                        _ => Err(()),
                    },
                    Some(Frame::Array) => match b {
                        b',' => {
                            self.mode = Mode::ArrVal { allow_close: false };
                            Ok(())
                        }
                        b']' => {
                            self.stack.pop();
                            self.after_close();
                            Ok(())
                        }
                        _ => Err(()),
                    },
                    None => Err(()),
                }
            }
            Mode::Done => {
                // After the top-level object, only trailing whitespace is valid.
                if is_ws(b) {
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }

    fn advance_num(&mut self, num: Num, b: u8) -> Result<(), ()> {
        // Complete sub-states end the number on a delimiter, which is then
        // re-processed in `AfterValue`.
        let complete = matches!(num, Num::Zero | Num::Int | Num::Frac | Num::Exp);
        if complete && is_num_delim(b) {
            self.mode = Mode::AfterValue;
            return self.advance(b);
        }
        let next = match num {
            Num::Sign => match b {
                b'0' => Num::Zero,
                b'1'..=b'9' => Num::Int,
                _ => return Err(()),
            },
            Num::Zero => match b {
                b'.' => Num::DotFirst,
                b'e' | b'E' => Num::ExpSign,
                _ => return Err(()), // no leading-zero digits, e.g. "01"
            },
            Num::Int => match b {
                _ if is_digit(b) => Num::Int,
                b'.' => Num::DotFirst,
                b'e' | b'E' => Num::ExpSign,
                _ => return Err(()),
            },
            Num::DotFirst => {
                if is_digit(b) {
                    Num::Frac
                } else {
                    return Err(());
                }
            }
            Num::Frac => match b {
                _ if is_digit(b) => Num::Frac,
                b'e' | b'E' => Num::ExpSign,
                _ => return Err(()),
            },
            Num::ExpSign => match b {
                b'+' | b'-' => Num::ExpFirst,
                _ if is_digit(b) => Num::Exp,
                _ => return Err(()),
            },
            Num::ExpFirst => {
                if is_digit(b) {
                    Num::Exp
                } else {
                    return Err(());
                }
            }
            Num::Exp => {
                if is_digit(b) {
                    Num::Exp
                } else {
                    return Err(());
                }
            }
        };
        self.mode = Mode::Num(next);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a full string; returns the final state if every byte was accepted.
    fn feed(s: &str) -> Result<JsonState, ()> {
        let mut st = JsonState::new();
        for &b in s.as_bytes() {
            st.advance(b)?;
        }
        Ok(st)
    }

    fn valid_complete(s: &str) -> bool {
        feed(s).map(|st| st.is_done()).unwrap_or(false)
    }

    fn valid_prefix(s: &str) -> bool {
        feed(s).is_ok()
    }

    #[test]
    fn accepts_complete_objects() {
        for s in [
            r#"{}"#,
            r#"{"a":1}"#,
            r#"{ "a" : 1 , "b" : 2 }"#,
            r#"{"a":[1,2,3]}"#,
            r#"{"a":{"b":{"c":[]}}}"#,
            r#"{"s":"hi \"there\"\né"}"#,
            r#"{"n":-12.5e+10,"z":0,"f":false,"t":true,"x":null}"#,
            "{\n  \"k\": 1\n}\n",
        ] {
            assert!(valid_complete(s), "should be complete valid: {s}");
            // serde agrees it is real JSON.
            assert!(serde_json::from_str::<serde_json::Value>(s.trim()).is_ok());
        }
    }

    #[test]
    fn rejects_invalid_bytes() {
        // Each must be rejected at some byte (never a valid prefix all the way).
        for s in [
            "[1]",         // top level must be an object
            "{,}",         // expected key
            "{\"a\":}",    // value required
            "{\"a\":1,}",  // trailing comma -> needs a key
            "{\"a\" 1}",   // missing colon
            "{\"a\":01}",  // leading zero
            "{\"a\":1.}",  // dangling fraction is not complete... but as a prefix it's ok
            "{\"a\":tru}", // bad literal
            "{'a':1}",     // single quotes
        ] {
            // `1.}` is a valid *prefix* until the `}`; assert it is not *complete*.
            assert!(!valid_complete(s), "should not be complete valid JSON: {s}");
        }
        assert!(!valid_prefix("[1]"));
        assert!(!valid_prefix("{,}"));
        assert!(!valid_prefix("{\"a\" 1}"));
        assert!(!valid_prefix("{\"a\":01}"));
        assert!(!valid_prefix("{'a':1}"));
    }

    #[test]
    fn tracks_prefix_validity() {
        // Genuine prefixes of a valid object are accepted but not complete.
        for p in [
            "",
            "{",
            "{\"",
            "{\"a",
            "{\"a\"",
            "{\"a\":",
            "{\"a\":1",
            "{\"a\":1,",
            "{\"a\":-",
            "{\"a\":1.",
            "{\"a\":1e",
            "{\"a\":[",
            "{\"a\":[1,",
            "{\"a\":tru",
        ] {
            assert!(valid_prefix(p), "should be a valid prefix: {p:?}");
            assert!(!valid_complete(p), "prefix should not be complete: {p:?}");
        }
    }

    #[test]
    fn accepts_masks_continuations_without_mutation() {
        let mut st = JsonState::new();
        for &b in br#"{"a":"# {
            st.advance(b).unwrap();
        }
        // A value is required next: these start valid values; `}`/`,` do not.
        assert!(st.accepts(b"1"));
        assert!(st.accepts(b"\"x\""));
        assert!(st.accepts(b"{"));
        assert!(st.accepts(b"true"));
        assert!(!st.accepts(b"}"));
        assert!(!st.accepts(b","));
        // accepts() must not have mutated the live state.
        assert!(!st.is_done());
        st.advance(b'1').unwrap();
        // After the value: ',' or '}' continue/close; another digit does not start.
        assert!(st.accepts(b","));
        assert!(st.accepts(b"}"));
        assert!(!st.accepts(b"x"));
    }

    #[test]
    fn done_only_after_top_level_close() {
        let mut st = JsonState::new();
        for &b in br#"{"a":1}"# {
            st.advance(b).unwrap();
        }
        assert!(st.is_done());
        // Trailing whitespace stays done; anything else is rejected.
        assert!(st.accepts(b"  \n"));
        assert!(!st.accepts(b"{"));
        assert!(!st.accepts(b"1"));
    }

    #[test]
    fn multibyte_tokens_validated_whole() {
        // A token whose bytes span structure ("}," or "1}") must validate as a unit.
        let mut st = JsonState::new();
        for &b in br#"{"a":[1"# {
            st.advance(b).unwrap();
        }
        assert!(st.accepts(b"]}")); // close array then object
        assert!(st.accepts(b",2")); // continue the array
        assert!(!st.accepts(b"}")); // can't close the object while inside the array
    }
}
