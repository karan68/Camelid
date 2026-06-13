//! `camelid pull` — fetch a supported model so the user never has to hunt for a
//! compatible GGUF by hand.
//!
//! Camelid only serves specific Q8_0 (and one Q4_0 QAT) rows, and most GGUFs on
//! the web are other quantizations that fail closed. This command downloads one
//! of the known-good rows from the same curated catalog the web UI uses, into
//! `./models`, and prints the exact `camelid serve` command to run next.

use std::path::{Path, PathBuf};

use crate::api::{curated_catalog, CatalogItem};

/// Entry point for the `Pull` subcommand. With no `query`, prints the catalog;
/// otherwise resolves `query` to exactly one row and downloads it into
/// `models_dir`.
pub fn run_pull(query: Option<&str>, models_dir: &Path) -> anyhow::Result<()> {
    let entries = curated_catalog();

    let Some(query) = query else {
        print_catalog(&entries);
        eprintln!("\nDownload one with:  camelid pull <id>   (e.g. camelid pull llama32_3b)");
        return Ok(());
    };

    let item = resolve(&entries, query)?;
    let dest = download(&item, models_dir)?;

    eprintln!("\n✓ {} is ready at {}", item.name, dest.display());
    eprintln!(
        "\nStart chatting:\n  camelid serve --model {}",
        dest.display()
    );
    Ok(())
}

/// Match `query` against the catalog by id or name, ignoring case and the
/// `-`/`_`/`.`/space separators people mix up. Errors helpfully on no match or
/// an ambiguous match rather than guessing.
fn resolve(entries: &[CatalogItem], query: &str) -> anyhow::Result<CatalogItem> {
    let needle = normalize(query);
    let matches: Vec<&CatalogItem> = entries
        .iter()
        .filter(|item| {
            normalize(item.catalog_id).contains(&needle) || normalize(item.name).contains(&needle)
        })
        .collect();

    match matches.as_slice() {
        [] => {
            print_catalog(entries);
            anyhow::bail!(
                "no supported model matches \"{query}\" — pick an id from the list above"
            );
        }
        [only] => Ok((*only).clone()),
        many => {
            let ids: Vec<&str> = many.iter().map(|item| item.catalog_id).collect();
            anyhow::bail!(
                "\"{query}\" matches several models ({}); be more specific",
                ids.join(", ")
            );
        }
    }
}

/// Download `item` into `models_dir` via `curl` (resumable, streaming progress
/// to the terminal). Skips the download if a complete copy already exists.
fn download(item: &CatalogItem, models_dir: &Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(models_dir)?;
    let dest = models_dir.join(item.filename);

    if let Ok(meta) = std::fs::metadata(&dest) {
        if meta.len() == item.size_bytes {
            eprintln!("{} already downloaded at {}", item.name, dest.display());
            return Ok(dest);
        }
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        item.repo_id, item.filename
    );
    eprintln!(
        "Downloading {} ({:.1} GB) from {}",
        item.name,
        item.size_bytes as f64 / 1e9,
        item.repo_id
    );

    // -L follow redirects, -C - resume a partial file, --fail surface HTTP
    // errors as a non-zero exit. curl writes its own progress bar to stderr.
    let status = std::process::Command::new("curl")
        .args(["-L", "-C", "-", "--fail", "-o"])
        .arg(&dest)
        .arg(&url)
        .status()
        .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;

    if !status.success() {
        anyhow::bail!("download failed (curl exited with {status}); re-run to resume");
    }
    Ok(dest)
}

fn print_catalog(entries: &[CatalogItem]) {
    eprintln!("Supported models (download into ./models):\n");
    eprintln!("  {:<28} {:<8} {:>8}  NAME", "ID", "QUANT", "SIZE");
    for item in entries {
        eprintln!(
            "  {:<28} {:<8} {:>6.1} GB  {}",
            item.catalog_id,
            item.quant,
            item.size_bytes as f64 / 1e9,
            item.name,
        );
    }
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | '.' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_by_id_fragment_ignoring_separators() {
        let entries = curated_catalog();
        let item = resolve(&entries, "llama32-3b").unwrap();
        assert_eq!(item.catalog_id, "llama32_3b_instruct_q8_0");
    }

    #[test]
    fn resolves_by_name_fragment() {
        let entries = curated_catalog();
        let item = resolve(&entries, "tinyllama").unwrap();
        assert!(item.catalog_id.contains("tinyllama"));
    }

    #[test]
    fn unknown_query_is_an_error() {
        let entries = curated_catalog();
        assert!(resolve(&entries, "gpt-9-turbo").is_err());
    }
}
