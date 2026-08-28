//! Bounded server-side web research for the embedded Chat UI.
//!
//! This is deliberately a pre-generation data path, not a model tool. Gemma 4
//! rows do not have a certified tool-call renderer, while the Web UI still
//! needs to ground a prompt that contains links or explicitly asks for current
//! information. The browser asks this endpoint first and injects the returned,
//! explicitly-untrusted excerpts into the ordinary chat request.
//!
//! Network safety is owned here. Only credential-free public HTTP(S) targets
//! on the default ports are accepted. Every redirect is parsed, checked, and
//! DNS-resolved again before curl sees it; validated DNS answers are pinned with
//! `--resolve`, proxy use and curlrc loading are disabled, and response/time/
//! redirect counts are bounded. Incoming Camelid authentication headers never
//! enter this module and therefore cannot be forwarded to the public web.

use std::{
    collections::{HashMap, HashSet},
    io::Read,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs},
    process::{Command, Stdio},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use url::{Host, Url};

use super::{api_error, AppState};

const MAX_PROMPT_BYTES: usize = 64 * 1024;
const MAX_DIRECT_URLS: usize = 4;
const MAX_SEARCH_RESULTS: usize = 8;
const MAX_SEARCH_FETCHES: usize = 3;
const MAX_REDIRECTS: usize = 4;
const MAX_HTTP_BODY_BYTES: usize = 512 * 1024;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_SOURCE_EXCERPT_CHARS: usize = 12_000;
const MAX_README_CHARS: usize = 4_000;
const MAX_CODE_FILE_CHARS: usize = 2_400;
const MAX_GITHUB_SOURCE_FILES: usize = 3;
const MAX_SEARCH_QUERY_CHARS: usize = 600;
const MAX_CONCURRENT_RESEARCH: usize = 2;
const RESEARCH_ADMISSION_TIMEOUT: Duration = Duration::from_millis(750);
const RESEARCH_TOTAL_DEADLINE: Duration = Duration::from_secs(30);
const BRAVE_SEARCH_ENDPOINT: &str = "https://search.brave.com/search";
const DDG_SEARCH_ENDPOINT: &str = "https://lite.duckduckgo.com/lite/";
const BING_SEARCH_ENDPOINT: &str = "https://www.bing.com/search";
const USER_AGENT: &str = "Camelid-WebResearch/0.6 (+https://github.com/timtoole02/Camelid)";

#[derive(Debug, Deserialize)]
pub(super) struct WebResearchRequest {
    prompt: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct WebResearchSource {
    pub id: usize,
    pub title: String,
    pub url: String,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct WebResearchResponse {
    /// complete | partial | skipped | failed
    pub status: &'static str,
    /// True whenever the prompt requested research, even if providers failed.
    pub triggered: bool,
    /// embedded_urls | explicit_search | current_info | not_needed
    pub reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    pub sources: Vec<WebResearchSource>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub(super) struct FetchResponse {
    status: u16,
    content_type: String,
    body: Vec<u8>,
    location: Option<String>,
    final_url: Option<Url>,
}

impl FetchResponse {
    #[cfg(test)]
    fn text(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: content_type.to_string(),
            body: body.into(),
            location: None,
            final_url: None,
        }
    }
}

/// Small injection seam: production uses curl, tests provide in-memory pages.
/// No request headers are accepted, which makes forwarding a Camelid API key
/// structurally impossible.
pub(super) trait WebTransport: Send + Sync {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String>;
}

#[derive(Default)]
struct CurlWebTransport;

struct DeadlineTransport<'a> {
    inner: &'a dyn WebTransport,
    deadline: Instant,
}

impl WebTransport for DeadlineTransport<'_> {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
        if Instant::now() >= self.deadline {
            return Err(format!(
                "the {}-second web research deadline was reached",
                RESEARCH_TOTAL_DEADLINE.as_secs()
            ));
        }
        self.inner.fetch(url, accept)
    }
}

fn research_semaphore() -> Arc<tokio::sync::Semaphore> {
    static SEMAPHORE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SEMAPHORE
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_RESEARCH)))
        .clone()
}

pub(super) fn default_transport() -> Arc<dyn WebTransport> {
    Arc::new(CurlWebTransport)
}

impl WebTransport for CurlWebTransport {
    fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
        let mut current = url.clone();
        for redirects in 0..=MAX_REDIRECTS {
            validate_url_shape(&current)?;
            let addresses = resolve_public_addresses(&current)?;
            let mut response = curl_single_hop(&current, accept, &addresses)?;
            if !(300..400).contains(&response.status) {
                response.final_url = Some(current);
                return Ok(response);
            }
            if redirects == MAX_REDIRECTS {
                return Err(format!("redirect limit ({MAX_REDIRECTS}) exceeded"));
            }
            let location = response
                .location
                .as_deref()
                .ok_or_else(|| format!("HTTP {} redirect omitted Location", response.status))?;
            current = validated_redirect_target(&current, location)?;
        }
        unreachable!("bounded redirect loop always returns")
    }
}

pub(super) async fn handler(
    State(state): State<AppState>,
    payload: Result<Json<WebResearchRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_web_research_json",
                format!("invalid /api/web/research JSON request: {error}"),
                Some("body"),
            )
        }
    };
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_web_research_prompt",
            "prompt must not be empty".to_string(),
            Some("prompt"),
        );
    }
    if prompt.len() > MAX_PROMPT_BYTES {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "web_research_prompt_too_large",
            format!("prompt exceeds the {MAX_PROMPT_BYTES}-byte web-research limit"),
            Some("prompt"),
        );
    }

    let decision = classify_prompt(prompt);
    if matches!(decision, ResearchDecision::Skip) {
        return Json(skipped_response()).into_response();
    }
    let reason = decision.reason();
    let permit = match tokio::time::timeout(
        RESEARCH_ADMISSION_TIMEOUT,
        research_semaphore().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) | Err(_) => {
            return Json(WebResearchResponse {
                status: "failed",
                triggered: true,
                reason,
                query: None,
                sources: Vec::new(),
                warnings: vec![
                    "Web research is busy; local chat can continue without web sources."
                        .to_string(),
                ],
            })
            .into_response()
        }
    };

    let prompt = prompt.to_string();
    let transport = state.web_research_transport.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Keep admission for the whole blocking job, even if the HTTP client
        // disconnects and Axum drops its waiter. This caps abandoned curl work.
        let _permit = permit;
        let bounded = DeadlineTransport {
            inner: transport.as_ref(),
            deadline: Instant::now() + RESEARCH_TOTAL_DEADLINE,
        };
        research_prompt(&prompt, &bounded)
    })
    .await
    .unwrap_or_else(|error| WebResearchResponse {
        status: "failed",
        triggered: true,
        reason,
        query: None,
        sources: Vec::new(),
        warnings: vec![format!("web research worker stopped unexpectedly: {error}")],
    });
    Json(result).into_response()
}

#[derive(Debug)]
enum ResearchDecision {
    Direct { urls: Vec<Url>, omitted: bool },
    Search { reason: &'static str, query: String },
    Skip,
}

impl ResearchDecision {
    fn reason(&self) -> &'static str {
        match self {
            Self::Direct { .. } => "embedded_urls",
            Self::Search { reason, .. } => reason,
            Self::Skip => "not_needed",
        }
    }
}

fn skipped_response() -> WebResearchResponse {
    WebResearchResponse {
        status: "skipped",
        triggered: false,
        reason: "not_needed",
        query: None,
        sources: Vec::new(),
        warnings: Vec::new(),
    }
}

fn research_prompt(prompt: &str, transport: &dyn WebTransport) -> WebResearchResponse {
    match classify_prompt(prompt) {
        ResearchDecision::Skip => skipped_response(),
        ResearchDecision::Direct { urls, omitted } => {
            let mut warnings = Vec::new();
            if omitted {
                warnings.push(format!(
                    "Only the first {MAX_DIRECT_URLS} distinct URLs were researched."
                ));
            }
            let mut sources = Vec::new();
            for url in urls {
                fetch_source(transport, &url, None, &mut sources, &mut warnings);
            }
            finish_response("embedded_urls", None, sources, warnings)
        }
        ResearchDecision::Search { reason, query } => search_and_fetch(transport, reason, query),
    }
}

fn finish_response(
    reason: &'static str,
    query: Option<String>,
    mut sources: Vec<WebResearchSource>,
    warnings: Vec<String>,
) -> WebResearchResponse {
    for (index, source) in sources.iter_mut().enumerate() {
        source.id = index + 1;
    }
    let status = if sources.is_empty() {
        "failed"
    } else if warnings.is_empty() {
        "complete"
    } else {
        "partial"
    };
    WebResearchResponse {
        status,
        triggered: true,
        reason,
        query,
        sources,
        warnings,
    }
}

fn classify_prompt(prompt: &str) -> ResearchDecision {
    let (urls, omitted) = extract_prompt_urls(prompt);
    if !urls.is_empty() {
        return ResearchDecision::Direct { urls, omitted };
    }

    let lower = prompt.to_lowercase();
    let explicit = [
        "search the web",
        "search web",
        "web search",
        "search online",
        "look this up",
        "look it up",
        "look up",
        "lookup",
        "look up online",
        "browse the web",
        "browse web",
        "browse the internet",
        "browse internet",
        "browse online",
        "use the internet",
        "find online",
        "find it online",
        "find this online",
        "find that online",
        "research online",
        "research this",
        "research that",
        "research on the web",
        "read the linked",
        "read linked",
        "read the website",
        "read website",
        "read the web page",
        "read web page",
        "read the github",
        "read github",
        "read the documentation",
        "read documentation",
        "read the docs",
        "read docs",
        "check the web",
        "check web",
        "check the internet",
        "check internet",
        "check the website",
        "check website",
        "check the github",
        "check github",
        "check the documentation",
        "check documentation",
        "check the docs",
        "check docs",
        "cite web sources",
        "cite online sources",
        "cite your sources",
        "cite your web sources",
        "cite your online sources",
        "cite sources",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let current = [
        "latest version",
        "latest release",
        "latest news",
        "latest price",
        "latest schedule",
        "latest score",
        "latest documentation",
        "latest docs",
        "latest specification",
        "latest status",
        "newest version",
        "newest release",
        "newest news",
        "newest price",
        "newest schedule",
        "newest score",
        "newest documentation",
        "newest docs",
        "newest specification",
        "newest status",
        "most recent",
        "current version",
        "current release",
        "current status",
        "current price",
        "current weather",
        "current schedule",
        "current score",
        "current documentation",
        "current docs",
        "current ceo",
        "currently available",
        "up-to-date",
        "up to date",
        "as of today",
        "as of now",
        "today's news",
        "today’s news",
        "today's weather",
        "today’s weather",
        "today's price",
        "today’s price",
        "today's schedule",
        "today’s schedule",
        "today's score",
        "today’s score",
        "recent news",
        "recent events",
        "recent developments",
        "recent changes",
        "recent updates",
        "recent releases",
        "news today",
        "weather today",
        "price today",
        "schedule today",
        "score today",
        "news right now",
        "weather right now",
        "price right now",
        "schedule right now",
        "score right now",
        "news now",
        "weather now",
        "price now",
        "schedule now",
        "score now",
        "current officeholder",
        "who is the current",
        "who is current",
        "what's new in",
        "what's new with",
        "what is new in",
        "what is new with",
        "what is the latest",
        "what's the latest",
        "what is the current",
        "what's the current",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || contains_as_of_year(&lower);

    if explicit || current {
        return ResearchDecision::Search {
            reason: if explicit {
                "explicit_search"
            } else {
                "current_info"
            },
            query: clip_chars(prompt.trim(), MAX_SEARCH_QUERY_CHARS, false),
        };
    }
    ResearchDecision::Skip
}

fn contains_as_of_year(text: &str) -> bool {
    text.match_indices("as of ").any(|(index, marker)| {
        let tail = &text[index + marker.len()..];
        let year = tail.as_bytes().get(..4);
        year.is_some_and(|digits| digits.iter().all(u8::is_ascii_digit))
            && tail
                .as_bytes()
                .get(4)
                .is_none_or(|next| !next.is_ascii_digit())
    })
}

fn extract_prompt_urls(prompt: &str) -> (Vec<Url>, bool) {
    let mut offsets = Vec::new();
    for scheme in ["https://", "http://"] {
        offsets.extend(prompt.match_indices(scheme).map(|(offset, _)| offset));
    }
    offsets.sort_unstable();

    let mut seen = HashSet::new();
    let mut urls = Vec::new();
    let mut omitted = false;
    for offset in offsets {
        let tail = &prompt[offset..];
        let ipv6_host_close = ipv6_host_closing_bracket(tail);
        let end = tail
            .char_indices()
            .skip(1)
            .find_map(|(index, ch)| {
                (ch.is_whitespace()
                    || matches!(ch, '<' | '>' | '"' | '\'' | '`')
                    || (ch == ']' && Some(index) != ipv6_host_close))
                    .then_some(index)
            })
            .unwrap_or(tail.len());
        let raw = tail[..end]
            .trim_end_matches(|ch: char| matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | '}'));
        let Ok(parsed) = Url::parse(raw) else {
            continue;
        };
        let normalized = canonical_prompt_url(parsed);
        let key = normalized.as_str().to_string();
        if !seen.insert(key) {
            continue;
        }
        if urls.len() == MAX_DIRECT_URLS {
            omitted = true;
            continue;
        }
        urls.push(normalized);
    }
    (urls, omitted)
}

fn ipv6_host_closing_bracket(candidate: &str) -> Option<usize> {
    let authority = candidate
        .strip_prefix("https://")
        .or_else(|| candidate.strip_prefix("http://"))?;
    if !authority.starts_with('[') {
        return None;
    }
    let relative = authority.find(']')?;
    Some(candidate.len() - authority.len() + relative)
}

fn canonical_prompt_url(mut url: Url) -> Url {
    url.set_fragment(None);
    // Never normalize away a security-relevant component. It must survive to
    // `validate_url_shape`, where it is rejected before any transport call.
    if url.username().is_empty() && url.password().is_none() && url.port().is_none() {
        if let Some(repo) = github_repo(&url) {
            return repo.canonical_url;
        }
    }
    // A Markdown destination commonly ends in one unmatched `)` followed by
    // punctuation. Preserve balanced/encoded parens inside real paths.
    let opens = url.path().chars().filter(|ch| *ch == '(').count();
    let closes = url.path().chars().filter(|ch| *ch == ')').count();
    if closes > opens && url.path().ends_with(')') {
        let next = url.path().trim_end_matches(')').to_string();
        url.set_path(&next);
    }
    url
}

#[derive(Debug, Clone)]
struct GithubRepo {
    owner: String,
    repo: String,
    canonical_url: Url,
}

fn github_repo(url: &Url) -> Option<GithubRepo> {
    if !url
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("github.com"))
    {
        return None;
    }
    let mut segments = url.path_segments()?.filter(|part| !part.is_empty());
    let owner = github_slug_prefix(segments.next()?);
    let repo = github_slug_prefix(segments.next()?).trim_end_matches(".git");
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    let canonical_url = Url::parse(&format!("https://github.com/{owner}/{repo}")).ok()?;
    Some(GithubRepo {
        owner: owner.to_string(),
        repo: repo.to_string(),
        canonical_url,
    })
}

fn github_slug_prefix(value: &str) -> &str {
    let end = value
        .char_indices()
        .find_map(|(index, ch)| {
            (!ch.is_ascii_alphanumeric() && !matches!(ch, '-' | '_' | '.')).then_some(index)
        })
        .unwrap_or(value.len());
    &value[..end]
}

fn fetch_source(
    transport: &dyn WebTransport,
    url: &Url,
    title_hint: Option<&str>,
    sources: &mut Vec<WebResearchSource>,
    warnings: &mut Vec<String>,
) {
    if let Err(error) = validate_url_shape(url) {
        warnings.push(format!("Blocked web URL: {error}"));
        return;
    }
    if let Some(repo) = github_repo(url) {
        if let Some(source) = fetch_github_repo(transport, &repo, warnings) {
            sources.push(source);
        }
        return;
    }
    let response = match transport.fetch(url, "text/html,text/plain,application/json;q=0.8") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!("Could not fetch {}: {error}", display_url(url)));
            return;
        }
    };
    if !(200..300).contains(&response.status) {
        warnings.push(format!(
            "Could not fetch {}: HTTP {}",
            display_url(url),
            response.status
        ));
        return;
    }
    let Some(excerpt) = response_excerpt(&response) else {
        warnings.push(format!(
            "Could not use {}: unsupported or empty content",
            display_url(url)
        ));
        return;
    };
    let provenance_url = response.final_url.as_ref().unwrap_or(url);
    let text = String::from_utf8_lossy(&response.body);
    let title = title_hint
        .filter(|title| !title.trim().is_empty())
        .map(str::to_string)
        .or_else(|| extract_html_title(&text))
        .unwrap_or_else(|| display_url(provenance_url));
    sources.push(WebResearchSource {
        id: 0,
        title: clip_chars(&title, 240, false),
        url: provenance_url.as_str().to_string(),
        excerpt,
    });
}

#[derive(Debug, Deserialize)]
struct GithubMetadata {
    default_branch: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GithubTree {
    #[serde(default)]
    tree: Vec<GithubTreeEntry>,
    #[serde(default)]
    truncated: bool,
}

#[derive(Debug, Deserialize)]
struct GithubTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    size: Option<u64>,
}

fn fetch_github_repo(
    transport: &dyn WebTransport,
    repo: &GithubRepo,
    warnings: &mut Vec<String>,
) -> Option<WebResearchSource> {
    let label = format!("{}/{}", repo.owner, repo.repo);
    let metadata_url = github_api_url(&repo.owner, &repo.repo, &[]);
    let branch = transport
        .fetch(&metadata_url, "application/vnd.github+json")
        .ok()
        .filter(|response| (200..300).contains(&response.status))
        .and_then(|response| serde_json::from_slice::<GithubMetadata>(&response.body).ok())
        .and_then(|metadata| metadata.default_branch)
        .filter(|branch| !branch.trim().is_empty())
        .unwrap_or_else(|| "HEAD".to_string());

    let mut readme = None;
    let readme_url = github_api_url(&repo.owner, &repo.repo, &["readme"]);
    match transport.fetch(&readme_url, "application/vnd.github.raw+json") {
        Ok(response) if (200..300).contains(&response.status) => {
            if let Some(text) = response_excerpt_with_limit(&response, MAX_README_CHARS) {
                readme = Some(text);
            } else {
                warnings.push(format!("{label}: README was empty or not text"));
            }
        }
        Ok(response) => warnings.push(format!(
            "{label}: README request returned HTTP {}",
            response.status
        )),
        Err(error) => warnings.push(format!("{label}: README unavailable: {error}")),
    }

    // Implementation evidence comes first so the browser's final, stricter
    // prompt budget cannot spend the whole allowance on README prose.
    let mut sections = Vec::new();
    let tree_url = github_api_url(&repo.owner, &repo.repo, &["git", "trees", &branch]);
    match transport.fetch(&tree_url, "application/vnd.github+json") {
        Ok(response) if (200..300).contains(&response.status) => {
            match serde_json::from_slice::<GithubTree>(&response.body) {
                Ok(tree) => {
                    if tree.truncated {
                        warnings.push(format!(
                            "{label}: GitHub returned a truncated file tree; only ranked visible files were considered"
                        ));
                    }
                    for path in select_github_source_paths(&tree.tree) {
                        let raw_url = raw_github_url(&repo.owner, &repo.repo, &branch, &path);
                        match transport.fetch(&raw_url, "text/plain") {
                            Ok(response) if (200..300).contains(&response.status) => {
                                if let Some(text) = code_excerpt(&response, MAX_CODE_FILE_CHARS) {
                                    sections.push((format!("Source: {path}"), text));
                                }
                            }
                            Ok(response) => warnings
                                .push(format!("{label}: {path} returned HTTP {}", response.status)),
                            Err(error) => {
                                warnings.push(format!("{label}: {path} unavailable: {error}"))
                            }
                        }
                    }
                }
                Err(error) => warnings.push(format!(
                    "{label}: GitHub file tree was not valid JSON: {error}"
                )),
            }
        }
        Ok(response) => warnings.push(format!(
            "{label}: GitHub file tree returned HTTP {}",
            response.status
        )),
        Err(error) => warnings.push(format!("{label}: GitHub file tree unavailable: {error}")),
    }

    let title = readme
        .as_deref()
        .and_then(markdown_title)
        .unwrap_or_else(|| label.clone());
    if let Some(readme) = readme {
        sections.push(("README".to_string(), readme));
    }
    if sections.is_empty() {
        warnings.push(format!(
            "Could not retrieve readable repository content for {label}"
        ));
        return None;
    }
    let combined = sections
        .into_iter()
        .map(|(heading, text)| format!("## {heading}\n{text}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    Some(WebResearchSource {
        id: 0,
        title: clip_chars(&title, 240, false),
        url: repo.canonical_url.as_str().to_string(),
        excerpt: clip_chars(&combined, MAX_SOURCE_EXCERPT_CHARS, true),
    })
}

fn github_api_url(owner: &str, repo: &str, suffix: &[&str]) -> Url {
    let mut url = Url::parse("https://api.github.com/").expect("static GitHub API URL");
    {
        let mut path = url.path_segments_mut().expect("GitHub API URL is a base");
        path.extend(["repos", owner, repo]);
        path.extend(suffix.iter().copied());
    }
    if suffix.first() == Some(&"trees") || suffix.get(1) == Some(&"trees") {
        url.query_pairs_mut().append_pair("recursive", "1");
    }
    url
}

fn raw_github_url(owner: &str, repo: &str, branch: &str, path: &str) -> Url {
    let mut url =
        Url::parse("https://raw.githubusercontent.com/").expect("static GitHub raw-content URL");
    {
        let mut segments = url
            .path_segments_mut()
            .expect("GitHub raw-content URL is a base");
        segments.extend([owner, repo, branch]);
        segments.extend(path.split('/'));
    }
    url
}

fn select_github_source_paths(entries: &[GithubTreeEntry]) -> Vec<String> {
    let mut candidates: Vec<(i32, &str)> = entries
        .iter()
        .filter(|entry| entry.kind == "blob")
        .filter(|entry| entry.size.unwrap_or(0) <= MAX_HTTP_BODY_BYTES as u64)
        .filter_map(|entry| {
            source_path_score(&entry.path).map(|score| (score, entry.path.as_str()))
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.to_lowercase().cmp(&right.1.to_lowercase()))
    });
    candidates
        .into_iter()
        .take(MAX_GITHUB_SOURCE_FILES)
        .map(|(_, path)| path.to_string())
        .collect()
}

fn source_path_score(path: &str) -> Option<i32> {
    let lower = path.to_ascii_lowercase();
    let basename = lower.rsplit('/').next()?;
    if lower.contains("/vendor/")
        || lower.contains("/node_modules/")
        || lower.contains("/dist/")
        || lower.contains("/target/")
        || basename.ends_with(".min.js")
        || matches!(basename, "package-lock.json" | "cargo.lock" | "license")
        || basename.starts_with("readme")
    {
        return None;
    }
    let extension = basename.rsplit_once('.').map(|(_, ext)| ext).unwrap_or("");
    if ![
        "c", "cc", "cpp", "css", "go", "h", "hpp", "html", "java", "js", "jsx", "kt", "m", "mm",
        "php", "py", "rb", "rs", "sh", "swift", "ts", "tsx", "xml", "yaml", "yml",
    ]
    .contains(&extension)
    {
        return None;
    }

    let exact = match basename {
        "script.js" => 10_000,
        "scan_ble.py" => 9_900,
        "retrieve_data.py" => 9_800,
        _ => 0,
    };
    let keywords = [
        "bluetooth",
        "ble",
        "scale",
        "weight",
        "scan",
        "retrieve",
        "device",
        "connect",
    ]
    .iter()
    .filter(|keyword| lower.contains(**keyword))
    .count() as i32;
    let entrypoint = ["main.", "app.", "index.", "script."]
        .iter()
        .any(|prefix| basename.starts_with(prefix));
    let depth = path.matches('/').count().min(20) as i32;
    Some(exact + keywords * 300 + i32::from(entrypoint) * 500 + (40 - depth))
}

fn search_and_fetch(
    transport: &dyn WebTransport,
    reason: &'static str,
    query: String,
) -> WebResearchResponse {
    let mut warnings = Vec::new();
    let mut hits = search_brave(transport, &query, &mut warnings);
    if hits.is_empty() {
        hits = search_ddg(transport, &query, &mut warnings);
    }
    if hits.is_empty() {
        hits = search_bing(transport, &query, &mut warnings);
    }
    if hits.is_empty() {
        return finish_response(reason, Some(query), Vec::new(), warnings);
    }
    let mut sources = Vec::new();
    for hit in hits.into_iter().take(MAX_SEARCH_FETCHES) {
        fetch_source(
            transport,
            &hit.url,
            Some(&hit.title),
            &mut sources,
            &mut warnings,
        );
    }
    finish_response(reason, Some(query), sources, warnings)
}

fn search_brave(
    transport: &dyn WebTransport,
    query: &str,
    warnings: &mut Vec<String>,
) -> Vec<SearchHit> {
    let mut url = Url::parse(BRAVE_SEARCH_ENDPOINT).expect("static Brave search endpoint");
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("source", "web");
    let response = match transport.fetch(&url, "text/html") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!(
                "Brave web search was unavailable ({error}); trying DuckDuckGo Lite."
            ));
            return Vec::new();
        }
    };
    let body = String::from_utf8_lossy(&response.body);
    if response.status == 202 || looks_like_brave_challenge(&body) {
        warnings.push(format!(
            "Brave web search challenged the automated request (HTTP {}); trying DuckDuckGo Lite.",
            response.status
        ));
        return Vec::new();
    }
    if response.status != 200 {
        warnings.push(format!(
            "Brave web search returned HTTP {}; trying DuckDuckGo Lite.",
            response.status
        ));
        return Vec::new();
    }
    let hits = parse_brave_results(&body);
    if hits.is_empty() {
        warnings.push(
            "Brave web search returned no usable web results; trying DuckDuckGo Lite.".to_string(),
        );
    }
    hits
}

fn search_ddg(
    transport: &dyn WebTransport,
    query: &str,
    warnings: &mut Vec<String>,
) -> Vec<SearchHit> {
    let mut url = Url::parse(DDG_SEARCH_ENDPOINT).expect("static DDG Lite endpoint");
    url.query_pairs_mut().append_pair("q", query);
    let response = match transport.fetch(&url, "text/html") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!("DuckDuckGo Lite was unavailable: {error}"));
            return Vec::new();
        }
    };
    let body = String::from_utf8_lossy(&response.body);
    if response.status == 202 || looks_like_ddg_challenge(&body) {
        warnings.push(format!(
            "DuckDuckGo Lite challenged the automated request (HTTP {}); no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    if response.status != 200 {
        warnings.push(format!(
            "DuckDuckGo Lite returned HTTP {}; no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    let hits = parse_ddg_results(&body);
    if hits.is_empty() {
        warnings.push("DuckDuckGo Lite returned no usable results.".to_string());
    }
    hits
}

fn search_bing(
    transport: &dyn WebTransport,
    query: &str,
    warnings: &mut Vec<String>,
) -> Vec<SearchHit> {
    let mut url = Url::parse(BING_SEARCH_ENDPOINT).expect("static Bing search endpoint");
    url.query_pairs_mut().append_pair("q", query);
    let response = match transport.fetch(&url, "text/html") {
        Ok(response) => response,
        Err(error) => {
            warnings.push(format!("Bing web search was unavailable: {error}"));
            return Vec::new();
        }
    };
    let body = String::from_utf8_lossy(&response.body);
    if response.status == 202 || looks_like_bing_challenge(&body) {
        warnings.push(format!(
            "Bing web search challenged the automated request (HTTP {}); no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    if response.status != 200 {
        warnings.push(format!(
            "Bing web search returned HTTP {}; no result was treated as evidence.",
            response.status
        ));
        return Vec::new();
    }
    let hits = parse_bing_results(&body);
    if hits.is_empty() {
        warnings.push("Bing web search returned no usable web results.".to_string());
    }
    hits
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchHit {
    title: String,
    url: Url,
}

fn looks_like_brave_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("<title>captcha")
        || lower.contains("<title>challenge")
        || lower.contains("cf-chl-")
        || lower.contains("scheduled a captcha")
        || lower.contains("suspicious request")
}

fn looks_like_ddg_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("captcha")
        || lower.contains("automated requests")
        || lower.contains("anomaly-modal")
        || lower.contains("challenge-form")
}

fn looks_like_bing_challenge(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("captcha")
        || lower.contains("unusual traffic")
        || lower.contains("verify you are human")
}

fn parse_brave_results(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    let markers = html
        .match_indices("data-type=\"web\"")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    for (position, marker) in markers.iter().copied().enumerate() {
        let start = html[..marker].rfind("<div").unwrap_or(marker);
        let end = markers.get(position + 1).copied().unwrap_or(html.len());
        let snippet = &html[start..end];
        let Some(anchor_start) = snippet.match_indices("<a ").find_map(|(index, _)| {
            let anchor = &snippet[index..];
            let tag_end = anchor.find('>')?;
            let attributes = &anchor[..tag_end];
            let classes = html_attribute(attributes, "class")?;
            classes
                .split_whitespace()
                .any(|class| class == "l1")
                .then_some(index)
        }) else {
            continue;
        };
        let anchor = &snippet[anchor_start..];
        let Some(tag_end) = anchor.find('>') else {
            continue;
        };
        let attributes = &anchor[..tag_end];
        let Some(href) = html_attribute(attributes, "href") else {
            continue;
        };
        let Some(url) = unwrap_search_redirect(&decode_html_entities(href)) else {
            continue;
        };
        let title = brave_result_title(snippet).unwrap_or_else(|| display_url(&url));
        if title.is_empty() || !seen.insert(url.as_str().to_string()) {
            continue;
        }
        hits.push(SearchHit { title, url });
        if hits.len() == MAX_SEARCH_RESULTS {
            break;
        }
    }
    hits
}

fn brave_result_title(snippet: &str) -> Option<String> {
    let marker = snippet.find("search-snippet-title")?;
    let tag_start = snippet[..marker].rfind('<')?;
    let tag_end = tag_start + snippet[tag_start..].find('>')?;
    let tag = &snippet[tag_start..tag_end];
    if let Some(title) = html_attribute(tag, "title") {
        let title = decode_html_entities(title).trim().to_string();
        if !title.is_empty() {
            return Some(title);
        }
    }
    let content_end = tag_end + 1 + snippet[tag_end + 1..].find("</")?;
    let title = strip_html(&snippet[tag_end + 1..content_end]);
    (!title.is_empty()).then_some(title)
}

fn html_attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    for quote in ['"', '\''] {
        let pattern = format!("{name}={quote}");
        if let Some((_, rest)) = tag.split_once(&pattern) {
            if let Some(end) = rest.find(quote) {
                return Some(&rest[..end]);
            }
        }
    }
    None
}

fn parse_ddg_results(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    for (index, _) in html.match_indices("result-link") {
        let before = &html[..index];
        let Some(anchor_at) = before.rfind("<a ") else {
            continue;
        };
        let anchor = &html[anchor_at..];
        let Some(tag_end) = anchor.find('>') else {
            continue;
        };
        let attributes = &anchor[..tag_end];
        let Some(href_at) = attributes.find("href=") else {
            continue;
        };
        let rest = &attributes[href_at + 5..];
        let Some(quote) = rest.chars().next().filter(|ch| matches!(ch, '\'' | '"')) else {
            continue;
        };
        let rest = &rest[quote.len_utf8()..];
        let Some(url_end) = rest.find(quote) else {
            continue;
        };
        let raw_href = decode_html_entities(&rest[..url_end]);
        let Some(url) = unwrap_search_redirect(&raw_href) else {
            continue;
        };
        let title = strip_html(anchor[tag_end + 1..].split("</a>").next().unwrap_or(""));
        if title.is_empty() || !seen.insert(url.as_str().to_string()) {
            continue;
        }
        hits.push(SearchHit { title, url });
        if hits.len() == MAX_SEARCH_RESULTS {
            break;
        }
    }
    hits
}

fn parse_bing_results(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    for (marker, _) in html.match_indices("class=\"b_algo\"") {
        let start = html[..marker].rfind("<li").unwrap_or(marker);
        let tail = &html[start..];
        let end = tail.find("</li>").unwrap_or(tail.len());
        let snippet = &tail[..end];
        let Some(heading) = snippet.find("<h2") else {
            continue;
        };
        let heading = &snippet[heading..];
        let Some(anchor_start) = heading.find("<a ") else {
            continue;
        };
        let anchor = &heading[anchor_start..];
        let Some(tag_end) = anchor.find('>') else {
            continue;
        };
        let Some(href) = html_attribute(&anchor[..tag_end], "href") else {
            continue;
        };
        let Some(url) = unwrap_search_redirect(&decode_html_entities(href)) else {
            continue;
        };
        let title = strip_html(anchor[tag_end + 1..].split("</a>").next().unwrap_or(""));
        if title.is_empty() || !seen.insert(url.as_str().to_string()) {
            continue;
        }
        hits.push(SearchHit { title, url });
        if hits.len() == MAX_SEARCH_RESULTS {
            break;
        }
    }
    hits
}

fn unwrap_search_redirect(href: &str) -> Option<Url> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    };
    let parsed = Url::parse(&absolute).ok()?;
    if parsed.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("duckduckgo.com")
            || host.to_ascii_lowercase().ends_with(".duckduckgo.com")
    }) {
        if let Some((_, target)) = parsed.query_pairs().find(|(key, _)| key == "uddg") {
            return Url::parse(&target).ok().map(canonical_prompt_url);
        }
    }
    if parsed.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("bing.com") || host.to_ascii_lowercase().ends_with(".bing.com")
    }) && parsed.path().starts_with("/ck/")
    {
        let encoded = parsed
            .query_pairs()
            .find_map(|(key, value)| (key == "u").then_some(value.into_owned()))?;
        let payload = encoded.strip_prefix("a1")?;
        let decoded = URL_SAFE_NO_PAD.decode(payload.as_bytes()).ok()?;
        let target = String::from_utf8(decoded).ok()?;
        return Url::parse(&target).ok().map(canonical_prompt_url);
    }
    matches!(parsed.scheme(), "http" | "https").then(|| canonical_prompt_url(parsed))
}

fn response_excerpt(response: &FetchResponse) -> Option<String> {
    response_excerpt_with_limit(response, MAX_SOURCE_EXCERPT_CHARS)
}

fn response_excerpt_with_limit(response: &FetchResponse, max_chars: usize) -> Option<String> {
    if !supported_text_content_type(&response.content_type) || !is_probably_text(&response.body) {
        return None;
    }
    let body = String::from_utf8_lossy(&response.body);
    let text = if response.content_type.to_ascii_lowercase().contains("html") {
        strip_html(&body)
    } else {
        clean_plain_text(&body)
    };
    (!text.is_empty()).then(|| clip_chars(&text, max_chars, true))
}

fn code_excerpt(response: &FetchResponse, max_chars: usize) -> Option<String> {
    if !supported_text_content_type(&response.content_type) || !is_probably_text(&response.body) {
        return None;
    }
    let text = clean_plain_text(&String::from_utf8_lossy(&response.body));
    if text.is_empty() {
        return None;
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return Some(text);
    }

    // Long source files often put setup/UI code before the protocol parser.
    // Keep a short header (imports, UUIDs and constants), then center the rest
    // of the budget on the first high-signal implementation entry point.
    let anchor = [
        "function handleNotifications",
        "def parse_smartchef_payload",
        "async def scan_for_smartchef",
        "async def scan_by_name",
        "navigator.bluetooth.requestDevice",
        "manufacturer_data",
        "weightMSB",
        "weight_raw",
    ]
    .iter()
    .find_map(|needle| text.find(needle))
    .map(|byte_index| text[..byte_index].chars().count());
    let Some(anchor) = anchor.filter(|anchor| *anchor > 500) else {
        return Some(clip_chars(&text, max_chars, true));
    };

    let marker = "\n\n[earlier setup omitted; relevant implementation follows]\n";
    let marker_chars = marker.chars().count();
    let prefix_chars = 400.min(max_chars / 4);
    let window_chars = max_chars.saturating_sub(prefix_chars + marker_chars);
    let prefix = chars.iter().take(prefix_chars).collect::<String>();
    let window = chars
        .iter()
        .skip(anchor)
        .take(window_chars)
        .collect::<String>();
    Some(format!("{prefix}{marker}{window}"))
}

fn supported_text_content_type(content_type: &str) -> bool {
    let content_type = content_type.to_ascii_lowercase();
    content_type.is_empty()
        || content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript")
        || content_type.contains("x-sh")
}

fn is_probably_text(body: &[u8]) -> bool {
    if body.is_empty() || body.contains(&0) {
        return false;
    }
    let sample = &body[..body.len().min(32 * 1024)];
    let control_bytes = sample
        .iter()
        .filter(|byte| **byte < 0x20 && !matches!(**byte, b'\n' | b'\r' | b'\t' | 0x0c))
        .count();
    if control_bytes > sample.len() / 100 + 2 {
        return false;
    }
    let decoded = String::from_utf8_lossy(sample);
    let replacements = decoded.chars().filter(|ch| *ch == '\u{fffd}').count();
    replacements <= decoded.chars().count() / 50 + 1
}

fn clean_plain_text(value: &str) -> String {
    value
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\0', "")
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn strip_html(value: &str) -> String {
    let without_scripts = remove_html_element(value, "script");
    let without_styles = remove_html_element(&without_scripts, "style");
    let mut text = String::with_capacity(without_styles.len());
    let mut inside_tag = false;
    for ch in without_styles.chars() {
        match ch {
            '<' => {
                inside_tag = true;
                text.push(' ');
            }
            '>' => inside_tag = false,
            _ if !inside_tag => text.push(ch),
            _ => {}
        }
    }
    decode_html_entities(&text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn remove_html_element(value: &str, tag: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(value.len());
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find(&open) {
        let start = cursor + relative_start;
        out.push_str(&value[cursor..start]);
        let Some(relative_end) = lower[start..].find(&close) else {
            return out;
        };
        cursor = start + relative_end + close.len();
    }
    out.push_str(&value[cursor..]);
    out
}

fn extract_html_title(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let content_start = start + lower[start..].find('>')? + 1;
    let content_end = content_start + lower[content_start..].find("</title>")?;
    let title = strip_html(&value[content_start..content_end]);
    (!title.is_empty()).then_some(title)
}

fn markdown_title(value: &str) -> Option<String> {
    value.lines().find_map(|line| {
        let title = line.trim().strip_prefix("# ")?.trim();
        (!title.is_empty()).then(|| title.to_string())
    })
}

fn decode_html_entities(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn clip_chars(value: &str, max_chars: usize, marker: bool) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut clipped: String = value.chars().take(max_chars).collect();
    if marker {
        clipped.push_str("\n[excerpt truncated]");
    }
    clipped
}

fn display_url(url: &Url) -> String {
    let host = url.host_str().unwrap_or("web target");
    let path = url.path();
    if path == "/" || path.is_empty() {
        host.to_string()
    } else {
        format!("{host}{path}")
    }
}

fn validate_url_shape(url: &Url) -> Result<(), String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("only public http:// and https:// URLs are allowed".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URLs containing credentials are not allowed".to_string());
    }
    if url.port().is_some() {
        return Err("only the default HTTP and HTTPS ports are allowed".to_string());
    }
    match url.host() {
        Some(Host::Domain(host)) => {
            let lower = host.to_ascii_lowercase();
            if lower == "localhost"
                || lower.ends_with(".localhost")
                || lower.ends_with(".local")
                || lower.ends_with(".internal")
            {
                return Err("local and internal hostnames are not allowed".to_string());
            }
        }
        Some(Host::Ipv4(address)) => {
            if !is_public_ip(IpAddr::V4(address)) {
                return Err(
                    "loopback, private, link-local, and reserved IPs are not allowed".into(),
                );
            }
        }
        Some(Host::Ipv6(address)) => {
            if !is_public_ip(IpAddr::V6(address)) {
                return Err(
                    "loopback, private, link-local, and reserved IPs are not allowed".into(),
                );
            }
        }
        None => return Err("URL must include a host".to_string()),
    }
    Ok(())
}

fn resolve_public_addresses(url: &Url) -> Result<Vec<IpAddr>, String> {
    validate_url_shape(url)?;
    if let Some(ip) = match url.host() {
        Some(Host::Ipv4(address)) => Some(IpAddr::V4(address)),
        Some(Host::Ipv6(address)) => Some(IpAddr::V6(address)),
        _ => None,
    } {
        return Ok(vec![ip]);
    }
    let host = url
        .host_str()
        .ok_or_else(|| "URL must include a host".to_string())?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL has no HTTP(S) port".to_string())?;
    let mut addresses: Vec<IpAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("could not resolve public host {host}: {error}"))?
        .map(|address| address.ip())
        .collect();
    addresses.sort_unstable();
    addresses.dedup();
    if addresses.is_empty() {
        return Err(format!("public host {host} did not resolve"));
    }
    if addresses.iter().any(|address| !is_public_ip(*address)) {
        return Err(format!(
            "host {host} resolved to a loopback, private, link-local, or reserved address"
        ));
    }
    Ok(addresses)
}

fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_private()
        || address.is_link_local()
        || address.is_multicast()
        || address.is_broadcast()
        || a == 0
        || (a == 100 && (64..=127).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 88 && c == 99)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 240)
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4() {
        return is_public_ipv4(mapped);
    }
    let segments = address.segments();
    !(address.is_unspecified()
        || address.is_loopback()
        || address.is_multicast()
        // Public IPv6 global unicast is allocated from 2000::/3. Reject
        // unallocated/reserved literals even if a host has a local route.
        || (segments[0] & 0xe000) != 0x2000
        || (segments[0] & 0xfe00) == 0xfc00
        || (segments[0] & 0xffc0) == 0xfe80
        || (segments[0] & 0xffc0) == 0xfec0
        || (segments[0] == 0x0064
            && segments[1] == 0xff9b
            && ((segments[2] == 0 && segments[3] == 0 && segments[4] == 0 && segments[5] == 0)
                || segments[2] == 1))
        || (segments[0] == 0x0100 && segments[1] == 0 && segments[2] == 0 && segments[3] == 0)
        || segments[0] == 0x2002
        || (segments[0] == 0x2001 && segments[1] == 0)
        || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        || (segments[0] == 0x3fff && (segments[1] & 0xf000) == 0))
}

fn validated_redirect_target(current: &Url, location: &str) -> Result<Url, String> {
    let next = current
        .join(location.trim())
        .map_err(|error| format!("invalid redirect target: {error}"))?;
    validate_url_shape(&next)?;
    if current.scheme() == "https" && next.scheme() != "https" {
        return Err("HTTPS redirects may not downgrade to cleartext HTTP".to_string());
    }
    Ok(next)
}

fn curl_single_hop(url: &Url, accept: &str, addresses: &[IpAddr]) -> Result<FetchResponse, String> {
    let host = url
        .host_str()
        .ok_or_else(|| "URL must include a host".to_string())?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "URL has no HTTP(S) port".to_string())?;
    let mut command = Command::new("curl");
    // -q/--disable must be the first curl option to prevent ~/.curlrc from
    // injecting credentials, proxies, or broader protocols.
    command.args([
        "-q",
        "--silent",
        "--show-error",
        "--no-progress-meter",
        "--noproxy",
        "*",
        "--proto",
        "=http,https",
        "--max-redirs",
        "0",
        "--connect-timeout",
        "5",
        "--max-time",
        "15",
        "--max-filesize",
        &MAX_HTTP_BODY_BYTES.to_string(),
        "--user-agent",
        USER_AGENT,
        "--header",
        &format!("Accept: {accept}"),
        "--dump-header",
        "-",
    ]);
    if url
        .host()
        .is_some_and(|host| matches!(host, Host::Domain(_)))
    {
        for address in addresses {
            let pinned = match address {
                IpAddr::V4(address) => format!("{host}:{port}:{address}"),
                IpAddr::V6(address) => format!("{host}:{port}:[{address}]"),
            };
            command.args(["--resolve", &pinned]);
        }
    }
    command
        .arg("--")
        .arg(url.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start curl: {error}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "curl stdout was unavailable".to_string())?;
    let output_limit = MAX_HTTP_HEADER_BYTES + MAX_HTTP_BODY_BYTES + 1;
    let mut output = Vec::with_capacity(output_limit.min(64 * 1024));
    stdout
        .by_ref()
        .take(output_limit as u64)
        .read_to_end(&mut output)
        .map_err(|error| format!("could not read curl response: {error}"))?;
    if output.len() >= output_limit {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!(
            "response exceeded the {MAX_HTTP_BODY_BYTES}-byte body limit"
        ));
    }
    let finished = child
        .wait_with_output()
        .map_err(|error| format!("could not finish curl: {error}"))?;
    if !finished.status.success() {
        let detail = String::from_utf8_lossy(&finished.stderr);
        return Err(if detail.trim().is_empty() {
            format!("curl exited with {}", finished.status)
        } else {
            format!("curl failed: {}", detail.trim())
        });
    }
    parse_curl_response(&output)
}

fn parse_curl_response(output: &[u8]) -> Result<FetchResponse, String> {
    let mut cursor = 0;
    loop {
        if !output[cursor..].starts_with(b"HTTP/") {
            return Err("web response did not include an HTTP status line".to_string());
        }
        let (header_end, delimiter_len) = find_header_end(&output[cursor..])
            .ok_or_else(|| "web response headers were incomplete".to_string())?;
        if header_end > MAX_HTTP_HEADER_BYTES {
            return Err("web response headers exceeded the safety limit".to_string());
        }
        let absolute_end = cursor + header_end;
        let header = String::from_utf8_lossy(&output[cursor..absolute_end]);
        let mut lines = header.lines();
        let status = lines
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u16>().ok())
            .ok_or_else(|| "web response had an invalid HTTP status".to_string())?;
        let mut headers = HashMap::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let body_start = absolute_end + delimiter_len;
        // A proxy CONNECT or informational response can precede the actual
        // response. Proxy use is disabled, but tolerate the shape defensively.
        if body_start < output.len()
            && output[body_start..].starts_with(b"HTTP/")
            && (status < 200 || status == 200)
        {
            cursor = body_start;
            continue;
        }
        let body = output[body_start..].to_vec();
        if body.len() > MAX_HTTP_BODY_BYTES {
            return Err(format!(
                "response exceeded the {MAX_HTTP_BODY_BYTES}-byte body limit"
            ));
        }
        return Ok(FetchResponse {
            status,
            content_type: headers.get("content-type").cloned().unwrap_or_default(),
            location: headers.get("location").cloned(),
            body,
            final_url: None,
        });
    }
}

fn find_header_end(value: &[u8]) -> Option<(usize, usize)> {
    value
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4))
        .or_else(|| {
            value
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|index| (index, 2))
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;

    #[derive(Default)]
    struct FakeTransport {
        responses: Mutex<HashMap<String, Result<FetchResponse, String>>>,
        calls: Mutex<Vec<(String, String)>>,
    }

    impl FakeTransport {
        fn insert(&self, url: Url, response: FetchResponse) {
            self.responses
                .lock()
                .unwrap()
                .insert(url.to_string(), Ok(response));
        }

        fn calls(&self) -> Vec<String> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .map(|(url, _)| url.clone())
                .collect()
        }
    }

    impl WebTransport for FakeTransport {
        fn fetch(&self, url: &Url, accept: &str) -> Result<FetchResponse, String> {
            self.calls
                .lock()
                .unwrap()
                .push((url.to_string(), accept.to_string()));
            self.responses
                .lock()
                .unwrap()
                .get(url.as_str())
                .cloned()
                .unwrap_or_else(|| Err(format!("no fixture for {url}")))
        }
    }

    fn github_fixtures(fake: &FakeTransport, owner: &str, repo: &str, files: &[(&str, &str)]) {
        fake.insert(
            github_api_url(owner, repo, &[]),
            FetchResponse::text(
                200,
                "application/json",
                br#"{"default_branch":"main"}"#.to_vec(),
            ),
        );
        fake.insert(
            github_api_url(owner, repo, &["readme"]),
            FetchResponse::text(
                200,
                "text/plain",
                format!("# {repo} documentation\nBluetooth scale details."),
            ),
        );
        let tree = json!({
            "tree": files.iter().map(|(path, _)| json!({
                "path": path,
                "type": "blob",
                "size": 128,
            })).collect::<Vec<_>>(),
            "truncated": false,
        });
        fake.insert(
            github_api_url(owner, repo, &["git", "trees", "main"]),
            FetchResponse::text(200, "application/json", tree.to_string()),
        );
        for (path, body) in files {
            fake.insert(
                raw_github_url(owner, repo, "main", path),
                FetchResponse::text(200, "text/plain", body.as_bytes().to_vec()),
            );
        }
    }

    #[test]
    fn ordinary_prompt_skips_without_touching_transport() {
        let fake = FakeTransport::default();
        let response = research_prompt("Turn this checklist into ordered steps.", &fake);
        assert_eq!(response.status, "skipped");
        assert!(!response.triggered);
        assert_eq!(response.reason, "not_needed");
        assert!(response.sources.is_empty());
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn current_and_explicit_frontend_cues_classify_as_search() {
        for prompt in [
            "Look up the Xcode documentation",
            "Lookup the Xcode documentation",
            "Read the linked docs for this package",
            "Check the website for recent changes",
            "What is today's weather?",
            "Who is the current CEO?",
            "Show recent events and updates",
            "What's new with Xcode?",
            "Explain this release and cite your sources",
            "Summarize the state as of 1999",
        ] {
            assert!(
                matches!(classify_prompt(prompt), ResearchDecision::Search { .. }),
                "did not search: {prompt}"
            );
        }
        assert!(matches!(
            classify_prompt("Explain what happened as of 20 minutes ago"),
            ResearchDecision::Skip
        ));
    }

    #[test]
    fn expired_total_deadline_blocks_transport_and_degrades_to_warning() {
        let fake = FakeTransport::default();
        let bounded = DeadlineTransport {
            inner: &fake,
            deadline: Instant::now() - Duration::from_millis(1),
        };
        let response = research_prompt("Read https://example.com/reference", &bounded);
        assert_eq!(response.status, "failed");
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("deadline")));
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn malformed_prompt_github_url_is_canonicalized_without_losing_valid_digits() {
        let (urls, omitted) = extract_prompt_urls(
            "Read https://github.com/PanamaHitek/SmartScale)2 and https://github.com/acme/repo2.",
        );
        assert!(!omitted);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://github.com/PanamaHitek/SmartScale",
                "https://github.com/acme/repo2",
            ]
        );
    }

    #[test]
    fn prompt_url_scanner_distinguishes_markdown_brackets_from_ipv6_hosts() {
        let (urls, omitted) = extract_prompt_urls(
            "Compare [https://example.com/label](https://example.com/destination) with http://[2606:4700:4700::1111]/dns.",
        );
        assert!(!omitted);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://example.com/label",
                "https://example.com/destination",
                "http://[2606:4700:4700::1111]/dns",
            ]
        );
    }

    #[test]
    fn two_github_urls_fetch_readmes_and_ranked_implementation_files() {
        let fake = FakeTransport::default();
        let long_script = format!(
            r#"const SCALE_SERVICE_UUID = 0xfff0;
const SCALE_CHARACTERISTIC_UUID = 0xfff1;
const DECIMALS = {{ 0b000: 0, 0b010: 1, 0b100: 2 }};
{}
function handleNotifications(event) {{
  const value = new Uint8Array(event.target.value.buffer);
  const {{ 3: attributes, 5: weightMSB, 6: weightLSB }} = value;
  if (value.slice(1).reduce((sum, d) => sum ^ d) != 0) throw new Error("checksum");
  const decimals = DECIMALS[attributes & 0b00000110];
  let weight = ((weightMSB << 8) + weightLSB) / 10 ** decimals;
  const precision = decimals == 0 ? String(weight).length : String(weight).length - 1;
}}"#,
            "function unrelatedSetup() {{ return true; }}\n".repeat(180)
        );
        github_fixtures(
            &fake,
            "bburky",
            "smartchef-web-bluetooth",
            &[
                ("style.css", "body {}"),
                ("index.html", "<button>Connect</button>"),
                ("script.js", &long_script),
            ],
        );
        let scan_ble = r#"from bleak import BleakScanner
TARGET_NAME = "SC02"
async def scan_by_name():
    return await BleakScanner.discover(timeout=5.0)"#;
        let retrieve_data = r#"from bleak import BleakScanner
def parse_smartchef_payload(data: bytes):
    b04 = data[4]
    b05 = data[5]
    weight_raw = b04 * 256 + b05
    weight = weight_raw / 100.0
def detection_callback(device, adv_data):
    for _, payload in adv_data.manufacturer_data.items():
        print(parse_smartchef_payload(payload))"#;
        github_fixtures(
            &fake,
            "PanamaHitek",
            "SmartScale",
            &[
                ("src/main/resources/python/retrieve_data.py", retrieve_data),
                ("src/main/resources/python/scan_ble.py", scan_ble),
                (
                    "src/main/java/com/panama_hitek/SmartScale.java",
                    "class SmartScale {}",
                ),
            ],
        );

        let response = research_prompt(
            "Use https://github.com/bburky/smartchef-web-bluetooth/ and https://github.com/PanamaHitek/SmartScale)2 to plan the app.",
            &fake,
        );
        assert_eq!(response.status, "complete", "{:?}", response.warnings);
        assert_eq!(response.reason, "embedded_urls");
        assert_eq!(response.sources.len(), 2);
        assert_eq!(response.sources[0].id, 1);
        assert_eq!(
            response.sources[0].url,
            "https://github.com/bburky/smartchef-web-bluetooth"
        );
        assert!(response.sources[0].excerpt.contains("Source: script.js"));
        for marker in [
            "function handleNotifications",
            "weightMSB",
            "weightLSB",
            "checksum",
            "precision",
            "10 ** decimals",
        ] {
            assert!(
                response.sources[0].excerpt.contains(marker),
                "lost {marker}"
            );
        }
        assert!(
            response.sources[0].excerpt.find("Source: script.js")
                < response.sources[0].excerpt.find("## README")
        );
        assert_eq!(response.sources[1].id, 2);
        assert_eq!(
            response.sources[1].url,
            "https://github.com/PanamaHitek/SmartScale"
        );
        for marker in [
            "scan_ble.py",
            "retrieve_data.py",
            "SC02",
            "manufacturer_data",
            "b04 * 256 + b05",
            "weight_raw / 100.0",
        ] {
            assert!(
                response.sources[1].excerpt.contains(marker),
                "lost {marker}"
            );
        }
        assert!(
            response.sources[1].excerpt.find("scan_ble.py")
                < response.sources[1].excerpt.find("## README")
        );
        assert!(fake.calls().iter().all(|url| !url.contains("duckduckgo")));
    }

    #[test]
    fn explicit_search_fetches_ranked_result_page() {
        let fake = FakeTransport::default();
        let prompt = "Search the web for the current Rust borrow checker documentation";
        let mut search_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        search_url
            .query_pairs_mut()
            .append_pair("q", prompt)
            .append_pair("source", "web");
        fake.insert(
            search_url,
            FetchResponse::text(
                200,
                "text/html",
                include_str!("../../tests/fixtures/websearch/brave_two_results.html"),
            ),
        );
        let first =
            Url::parse("https://blog.logrocket.com/introducing-rust-borrow-checker/").unwrap();
        fake.insert(
            first,
            FetchResponse::text(
                200,
                "text/html",
                "<html><title>Borrow checker guide</title><body>Ownership keeps Rust code safe.</body></html>",
            ),
        );
        // The fixture has a second result. Let it fail to prove partial results
        // still survive rather than pretending the whole search was empty.
        let response = research_prompt(prompt, &fake);
        assert_eq!(response.reason, "explicit_search");
        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert!(response.sources[0]
            .excerpt
            .contains("Ownership keeps Rust code safe"));
        assert!(response.query.as_deref().unwrap().contains("current Rust"));
        assert!(fake.calls().iter().all(|url| !url.contains("duckduckgo")));
    }

    #[test]
    fn ddg_202_or_captcha_is_a_provider_warning_not_no_results() {
        let fake = FakeTransport::default();
        let prompt = "What is the latest Xcode release?";
        let mut brave_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        brave_url
            .query_pairs_mut()
            .append_pair("q", prompt)
            .append_pair("source", "web");
        fake.insert(
            brave_url,
            FetchResponse::text(503, "text/html", "temporarily unavailable"),
        );
        let mut search_url = Url::parse(DDG_SEARCH_ENDPOINT).unwrap();
        search_url.query_pairs_mut().append_pair("q", prompt);
        fake.insert(
            search_url,
            FetchResponse::text(
                202,
                "text/html",
                "<form class=challenge-form>CAPTCHA</form>",
            ),
        );
        let response = research_prompt(prompt, &fake);
        assert_eq!(response.status, "failed");
        assert!(response.triggered);
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("Brave") && warning.contains("503")));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("DuckDuckGo Lite") && warning.contains("challenged")));
        assert!(response
            .warnings
            .iter()
            .all(|warning| !warning.contains("no usable results")));
    }

    #[test]
    fn bing_fallback_survives_brave_and_ddg_challenges() {
        let fake = FakeTransport::default();
        let prompt = "What is the latest Xcode release?";
        let mut brave_url = Url::parse(BRAVE_SEARCH_ENDPOINT).unwrap();
        brave_url
            .query_pairs_mut()
            .append_pair("q", prompt)
            .append_pair("source", "web");
        fake.insert(
            brave_url,
            FetchResponse::text(
                200,
                "text/html",
                "<title>Brave Search</title><p>We scheduled a captcha for this suspicious request.</p>",
            ),
        );
        let mut ddg_url = Url::parse(DDG_SEARCH_ENDPOINT).unwrap();
        ddg_url.query_pairs_mut().append_pair("q", prompt);
        fake.insert(
            ddg_url,
            FetchResponse::text(
                202,
                "text/html",
                "<form class=challenge-form>CAPTCHA</form>",
            ),
        );
        let mut bing_url = Url::parse(BING_SEARCH_ENDPOINT).unwrap();
        bing_url.query_pairs_mut().append_pair("q", prompt);
        fake.insert(
            bing_url,
            FetchResponse::text(
                200,
                "text/html",
                r#"<ol><li class="b_algo"><h2><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=fixture&amp;u=a1aHR0cHM6Ly9kZXZlbG9wZXIuYXBwbGUuY29tL3hjb2RlLw&amp;ntb=1"><strong>Xcode</strong> - Apple Developer</a></h2></li></ol>"#,
            ),
        );
        let source_url = Url::parse("https://developer.apple.com/xcode/").unwrap();
        fake.insert(
            source_url,
            FetchResponse::text(
                200,
                "text/html",
                "<title>Xcode - Apple Developer</title><main>Current Xcode release details.</main>",
            ),
        );

        let response = research_prompt(prompt, &fake);
        assert_eq!(response.status, "partial");
        assert_eq!(response.sources.len(), 1);
        assert_eq!(
            response.sources[0].url,
            "https://developer.apple.com/xcode/"
        );
        assert!(response.sources[0].excerpt.contains("Current Xcode"));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("Brave") && warning.contains("challenged")));
        assert!(response
            .warnings
            .iter()
            .any(|warning| warning.contains("DuckDuckGo Lite") && warning.contains("challenged")));
    }

    #[test]
    fn unsafe_targets_are_rejected_before_transport() {
        let fake = FakeTransport::default();
        for prompt in [
            "Read http://127.0.0.1/admin",
            "Read http://169.254.169.254/latest/meta-data",
            "Read http://10.0.0.2/private",
            "Read http://192.88.99.1/relay",
            "Read https://user:secret@example.com/private",
            "Read https://user:secret@github.com/acme/repo",
            "Read https://github.com:444/acme/repo",
            "Read http://[::1]/private",
            "Read http://[64:ff9b:1::7f00:1]/private",
            "Read http://[3fff::1]/reserved",
            "Read http://[4000::1]/reserved",
            "Read http://[100::1]/discard",
            "Read http://[2001::1]/teredo",
            "Read http://[2002:7f00:1::]/six-to-four",
        ] {
            let response = research_prompt(prompt, &fake);
            assert_eq!(response.status, "failed", "{prompt}");
            assert!(response.sources.is_empty());
        }
        assert!(fake.calls().is_empty());
    }

    #[test]
    fn binary_and_non_http_responses_are_rejected() {
        assert!(validate_url_shape(&Url::parse("file:///etc/passwd").unwrap()).is_err());
        assert!(response_excerpt(&FetchResponse::text(
            200,
            "image/png",
            b"not really a png".to_vec(),
        ))
        .is_none());
        assert!(response_excerpt(&FetchResponse::text(
            200,
            "application/octet-stream",
            b"text-looking binary\0payload".to_vec(),
        ))
        .is_none());
    }

    #[test]
    fn redirect_target_is_revalidated() {
        let base = Url::parse("https://example.com/start").unwrap();
        let error = validated_redirect_target(&base, "http://192.168.1.2/secret").unwrap_err();
        assert!(error.contains("not allowed"));
        let safe = validated_redirect_target(&base, "/next").unwrap();
        assert_eq!(safe.as_str(), "https://example.com/next");
        let downgrade = validated_redirect_target(&base, "http://example.com/next").unwrap_err();
        assert!(downgrade.contains("downgrade"));
    }

    #[test]
    fn source_selection_prioritizes_relevant_code() {
        let entries = [
            ("docs/guide.md", "blob"),
            ("style.css", "blob"),
            ("script.js", "blob"),
            ("src/main/resources/python/retrieve_data.py", "blob"),
            ("src/main/resources/python/scan_ble.py", "blob"),
            ("node_modules/no.js", "blob"),
        ]
        .into_iter()
        .map(|(path, kind)| GithubTreeEntry {
            path: path.to_string(),
            kind: kind.to_string(),
            size: Some(100),
        })
        .collect::<Vec<_>>();
        assert_eq!(
            select_github_source_paths(&entries),
            [
                "script.js",
                "src/main/resources/python/scan_ble.py",
                "src/main/resources/python/retrieve_data.py",
            ]
        );
    }

    #[test]
    fn curl_response_parser_bounds_and_extracts_headers() {
        let response = parse_curl_response(
            b"HTTP/2 302\r\nlocation: https://example.com/next\r\ncontent-type: text/plain\r\n\r\nredirect",
        )
        .unwrap();
        assert_eq!(response.status, 302);
        assert_eq!(
            response.location.as_deref(),
            Some("https://example.com/next")
        );
        assert_eq!(response.body, b"redirect");
    }

    #[tokio::test]
    async fn api_route_returns_typed_sources_and_skips_irrelevant_prompts() {
        let fake = Arc::new(FakeTransport::default());
        let url = Url::parse("https://example.com/reference").unwrap();
        fake.insert(
            url,
            FetchResponse::text(
                200,
                "text/html",
                "<title>Reference</title><main>grounded fact marker</main>",
            ),
        );
        let state = AppState::default().with_web_research_transport_for_tests(fake.clone());
        let app = super::super::router_with_state(state);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/web/research")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"prompt":"Read https://example.com/reference"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["triggered"], true);
        assert_eq!(body["sources"][0]["url"], "https://example.com/reference");
        assert!(body["sources"][0]["excerpt"]
            .as_str()
            .unwrap()
            .contains("grounded fact marker"));

        let before = fake.calls().len();
        let skipped = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/web/research")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt":"Rewrite this sentence."}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(skipped.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&to_bytes(skipped.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["status"], "skipped");
        assert_eq!(body["triggered"], false);
        assert_eq!(fake.calls().len(), before);
    }
}
