//! Lossless greedy speculative decoding.
//!
//! Decode is memory-bandwidth bound: every sequential token costs a full
//! weight read. Speculation drafts k candidate tokens cheaply, then verifies
//! them in ONE batched forward through the target model
//! (`forward_greedy_verify_chunk`), so a single weight read can yield several
//! accepted tokens. Every emitted token is the target model's own greedy
//! argmax given the accepted prefix, so accepted output is the same token
//! stream vanilla greedy decode produces; rejected drafts are dropped by KV
//! rollback and never observable.
//!
//! Support boundary: speculation is a default-off serving optimization. It
//! makes no support claim, moves no release-ledger row, and byte-parity for a
//! given lane is asserted only by evidence (tests and parity receipts), never
//! by resemblance.
//!
//! Two drafters:
//! - [`NGramDrafter`] (prompt lookup): proposes the continuation of the most
//!   recent earlier occurrence of the current suffix. Zero extra weights;
//!   wins on repetitive/structured text, proposes nothing on novel text.
//! - [`ModelDrafter`]: a small model (same tokenizer) greedily drafts k
//!   tokens; the target verifies them in one pass.

use crate::inference::{LlamaInferenceSession, LlamaSampler};
use crate::Result;

/// Default drafted tokens per round for the n-gram drafter. The n-gram lookup
/// itself is nearly free, but each extra draft widens the batched verify GEMM, so
/// over-drafting wastes work on partial-acceptance text (code, prose) without
/// helping. Measured on a 3B Q8_0 GPU resident decode (RTX 3060): a draft count of
/// 5 (verify batch k=6) sits at the sweet spot — within ~1% of the maximum
/// repetitive-text speedup (~2.55x) while giving the best result on moderately
/// repetitive code (~1.20x), where 7 drafts regress to ~1.09x. Bounded by
/// `cuda_resident::MAX_VERIFY_K - 1`.
pub const DEFAULT_NGRAM_DRAFT_TOKENS: usize = 5;

/// Default drafted tokens per round for the draft-model drafter. Each draft
/// token costs a sequential forward through the draft model, so the window
/// stays shorter.
pub const DEFAULT_MODEL_DRAFT_TOKENS: usize = 5;

/// Count the longest accepted prefix: drafted tokens that equal the target's
/// own greedy predictions position by position.
pub fn accepted_draft_prefix(drafts: &[u32], target_predictions: &[u32]) -> usize {
    drafts
        .iter()
        .zip(target_predictions.iter())
        .take_while(|(draft, prediction)| draft == prediction)
        .count()
}

pub enum SpeculativeDrafter {
    NGram(NGramDrafter),
    Model(Box<ModelDrafter>),
}

impl SpeculativeDrafter {
    /// Propose up to `max_tokens` draft tokens to follow `history` (the full
    /// token sequence so far: prompt plus generated, including the trailing
    /// token the target has not consumed yet). May return fewer or none.
    pub fn draft(&mut self, history: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        match self {
            Self::NGram(drafter) => Ok(drafter.draft(history, max_tokens)),
            Self::Model(drafter) => drafter.draft(history, max_tokens),
        }
    }
}

/// Prompt-lookup drafting: find the longest n-gram suffix of `history`
/// (between `min_ngram` and `max_ngram`) that occurred earlier, preferring
/// the most recent occurrence, and propose the tokens that followed it.
#[derive(Debug, Clone)]
pub struct NGramDrafter {
    pub max_ngram: usize,
    pub min_ngram: usize,
}

impl Default for NGramDrafter {
    fn default() -> Self {
        Self {
            max_ngram: 4,
            // Two-token patterns (e.g. ", " pairs) recur with unrelated
            // continuations and mostly waste verify rows; three-token
            // matches measure far higher acceptance.
            min_ngram: 3,
        }
    }
}

impl NGramDrafter {
    pub fn draft(&self, history: &[u32], max_tokens: usize) -> Vec<u32> {
        if max_tokens == 0 || self.min_ngram == 0 || history.len() <= self.min_ngram {
            return Vec::new();
        }
        let len = history.len();
        let max_n = self.max_ngram.min(len.saturating_sub(1));
        for n in (self.min_ngram..=max_n).rev() {
            let pattern = &history[len - n..];
            // Most recent earlier occurrence; the window at len-n is the
            // suffix itself and is excluded.
            for start in (0..len - n).rev() {
                if &history[start..start + n] == pattern {
                    let continuation_start = start + n;
                    let continuation_end = (continuation_start + max_tokens).min(len);
                    if continuation_start < continuation_end {
                        return history[continuation_start..continuation_end].to_vec();
                    }
                    break;
                }
            }
        }
        Vec::new()
    }
}

/// Draft-model drafting: a smaller model with the SAME token mapping runs
/// greedy decode ahead of the target. The draft session mirrors the accepted
/// sequence by re-ingesting tokens from `history` (`committed` counts the
/// history tokens whose KV entries are valid); each round's speculative tail
/// is rolled back before the next round, so rejected drafts never contaminate
/// the draft context.
pub struct ModelDrafter {
    session: LlamaInferenceSession,
    committed: usize,
    /// Drafted tokens fed into the session's KV beyond `committed` last
    /// round. The prefix of these that the target accepted is now real
    /// history, so its KV entries can be kept instead of re-ingested.
    speculative_fed: Vec<u32>,
}

impl ModelDrafter {
    pub fn new(mut session: LlamaInferenceSession) -> Self {
        // Route the draft session's GPU resident engine to the dedicated drafter
        // cache so it coexists with the target's engine. Resident decode stays
        // enabled (the draft model runs fast on the GPU); rollback of rejected
        // drafts uses `rollback_resident_to_position`, which resets the engine's
        // `filled` so the GPU KV (still valid up to the accepted prefix) is trusted
        // rather than reseeded. If the draft engine doesn't fit in VRAM it falls
        // back to the CPU path per token automatically.
        session.set_is_drafter(true);
        // Register the draft's resident VRAM footprint so a target engine built AFTER this
        // (e.g. when the drafter is configured before the target's first decode) leaves room for
        // the draft to stay GPU-resident too. Only honored on a GPU where the target still fits
        // fully resident after the reserve; otherwise the draft falls back to CPU. No-op on
        // non-CUDA builds. (Does not evict an already-built target — see set_spec_coexist_reserve.)
        crate::inference::set_spec_coexist_reserve(session.spec_coexist_reserve_estimate());
        Self {
            session,
            committed: 0,
            speculative_fed: Vec::new(),
        }
    }

    pub fn draft(&mut self, history: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        if max_tokens == 0 || history.is_empty() {
            return Ok(Vec::new());
        }
        // Last round's speculative KV entries start at `committed`. The
        // prefix that matches history (accepted drafts) is kept; only the
        // rejected tail is rolled back and never re-fed.
        let reuse = history[self.committed..]
            .iter()
            .zip(self.speculative_fed.iter())
            .take_while(|(token, fed)| token == fed)
            .count();
        self.session
            .rollback_resident_to_position(self.committed + reuse)?;
        self.committed += reuse;
        self.speculative_fed.clear();
        let pending = &history[self.committed..];
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        // Cap so the pending chunk plus the drafted tail fits the draft
        // model's context window.
        let room = self
            .session
            .remaining_context()
            .saturating_sub(pending.len());
        let max_tokens = max_tokens.min(room.saturating_add(1));
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        // Re-ingest the pending (known) tokens, then the prediction after the LAST one is the
        // first draft. The whole chunk rides the fast resident GPU-argmax lane (the draft only
        // needs the argmax, so the full-logits copy + CPU sample the diagnostics path does is pure
        // per-round overhead — the dominant cost once the draft model is GPU-resident). The
        // diagnostics path is the fallback only when the resident engine isn't ready (not yet
        // seeded), in which case nothing has been fed so re-feeding the whole chunk is consistent.
        // Lossless either way — the target verify is authoritative, so the draft's greedy choice
        // only affects accept rate, never the emitted tokens.
        let (&head, rest) = pending.split_first().expect("pending is non-empty (checked above)");
        let first = match self.session.generate_next_token_greedy_resident(head)? {
            Some((mut pred, _us)) => {
                for &tok in rest {
                    pred = match self.session.generate_next_token_greedy_resident(tok)? {
                        Some((id, _us)) => id,
                        // Residency is stable within a round; if it drops, feed just this one
                        // token via the general step so the KV stays consistent (one token each).
                        None => {
                            self.session
                                .generate_next_token_with_history_diagnostics(
                                    &[tok],
                                    LlamaSampler::Greedy,
                                    history,
                                    false,
                                )?
                                .next_token_id
                        }
                    };
                }
                pred
            }
            None => {
                self.session
                    .generate_next_token_with_history_diagnostics(
                        pending,
                        LlamaSampler::Greedy,
                        history,
                        false,
                    )?
                    .next_token_id
            }
        };
        self.committed = history.len();
        let mut drafts = Vec::with_capacity(max_tokens);
        drafts.push(first);
        while drafts.len() < max_tokens {
            let last = *drafts.last().expect("drafts is non-empty");
            // Sequential draft steps on the fast resident GPU-argmax lane (no full-logits copy).
            let next = match self.session.generate_next_token_greedy_resident(last)? {
                Some((id, _us)) => id,
                None => {
                    self.session
                        .generate_next_token_with_history_diagnostics(
                            &[last],
                            LlamaSampler::Greedy,
                            history,
                            false,
                        )?
                        .next_token_id
                }
            };
            drafts.push(next);
        }
        // KV now holds `committed` history tokens plus the fed drafts (all
        // but the last drafted token); the next round keeps whatever prefix
        // the target accepts and rolls back the rest.
        self.speculative_fed = drafts[..drafts.len().saturating_sub(1)].to_vec();
        Ok(drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_drafts_continuation_of_most_recent_match() {
        let drafter = NGramDrafter::default();
        // ... 1 2 3 4 | 9 9 | 1 2 3 4 | 7 8 | ... suffix [7, 8] has no
        // earlier match; suffix ending [3, 4] repeats.
        let history = vec![1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4];
        // Suffix [1, 2, 3, 4] (n=4) matches at start; continuation is [5, 6, 9].
        assert_eq!(drafter.draft(&history, 3), vec![5, 6, 9]);
    }

    #[test]
    fn ngram_prefers_longer_patterns_and_recent_matches() {
        let drafter = NGramDrafter::default();
        // [3, 4] occurs twice earlier with different continuations; the most
        // recent occurrence (followed by 8) wins.
        let history = vec![3, 4, 7, 0, 3, 4, 8, 0, 3, 4];
        assert_eq!(drafter.draft(&history, 2), vec![8, 0]);
    }

    #[test]
    fn ngram_returns_empty_when_no_repeat_exists() {
        let drafter = NGramDrafter::default();
        assert!(drafter.draft(&[1, 2, 3, 4, 5], 4).is_empty());
        assert!(drafter.draft(&[1, 2], 4).is_empty());
        assert!(drafter.draft(&[], 4).is_empty());
    }

    #[test]
    fn ngram_caps_at_requested_tokens() {
        let drafter = NGramDrafter {
            max_ngram: 3,
            min_ngram: 2,
        };
        let history = vec![1, 2, 9, 8, 7, 6, 1, 2];
        assert_eq!(drafter.draft(&history, 2), vec![9, 8]);
        assert_eq!(drafter.draft(&history, 10), vec![9, 8, 7, 6, 1, 2]);
    }

    #[test]
    fn accepted_prefix_counts_matches_until_first_divergence() {
        assert_eq!(accepted_draft_prefix(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(accepted_draft_prefix(&[1, 2, 3], &[1, 9, 3, 4]), 1);
        assert_eq!(accepted_draft_prefix(&[1, 2, 3], &[9, 9, 9, 9]), 0);
        assert_eq!(accepted_draft_prefix(&[], &[5]), 0);
    }
}
