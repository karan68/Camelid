//! Checkpoints: snapshot a file before the agent changes it, so the user can
//! see and undo what it did.
//!
//! Content snapshots only — never a `git stash`. Two reasons (DECISIONS
//! D-DROVER-5): one code path is easier to prove against the sandbox jail than
//! two, and the agent must not mutate git state the sandbox does not own. A
//! workspace that is not a repo, or is a repo with staged work the user cares
//! about, behaves identically.
//!
//! Snapshots live under `.camelid/checkpoints/` inside the workspace and are
//! resolved through the same canonical-prefix check as every other path, so a
//! checkpoint can neither be written nor restored outside the sandbox root.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use super::tools::Sandbox;

const DIR: &str = ".camelid/checkpoints";

/// One saved file state, taken immediately before a write.
#[derive(Clone)]
pub struct Checkpoint {
    /// Workspace-relative path of the file that was about to change.
    pub rel: String,
    /// Where the previous contents were saved, or `None` if the file did not
    /// exist yet (undo therefore means "delete it again").
    pub backup: Option<PathBuf>,
    pub tool: String,
}

/// Whether `path` resolves inside the sandbox root. Canonicalises the parent so
/// a symlinked final component cannot smuggle the target outside.
fn inside_root(sandbox: &Sandbox, path: &Path) -> bool {
    let Ok(root) = std::fs::canonicalize(sandbox.root()) else {
        return false;
    };
    let canon = match std::fs::canonicalize(path) {
        Ok(c) => c,
        Err(_) => {
            // Does not exist yet: resolve its parent instead.
            let Some(parent) = path.parent() else {
                return false;
            };
            let Ok(p) = std::fs::canonicalize(parent) else {
                return false;
            };
            match path.file_name() {
                Some(f) => p.join(f),
                None => return false,
            }
        }
    };
    canon.starts_with(&root)
}

fn log() -> &'static Mutex<Vec<Checkpoint>> {
    static L: OnceLock<Mutex<Vec<Checkpoint>>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn clear() {
    if let Ok(mut g) = log().lock() {
        g.clear();
    }
}

pub fn all() -> Vec<Checkpoint> {
    log().lock().map(|g| g.clone()).unwrap_or_default()
}

/// Snapshot `path` before it is written.
///
/// Best-effort by design: a failure to snapshot must not block the edit the
/// user asked for, so problems are reported and swallowed. The one thing it
/// will not do is write outside the jail.
pub fn take(sandbox: &Sandbox, path: &Path, tool: &str) {
    let raw = path.to_string_lossy();
    let resolved = match sandbox.resolve(&raw, path.exists()) {
        Ok(path) => path,
        Err(_) => return,
    };
    // Only files inside the workspace are snapshotted. With --allow-fs the
    // agent may legitimately write elsewhere, but copying those into an
    // in-workspace store would pull outside content across the boundary the
    // store lives behind — so they get no undo, rather than a leak.
    if !inside_root(sandbox, &resolved) {
        return;
    }
    let rel = sandbox.rel(&resolved);
    // Create the store first, then resolve it through the jail. `resolve` with
    // must_exist=false canonicalises the *parent*, so it cannot resolve a
    // two-level path whose first level does not exist yet — the store has to
    // exist before it can be checked, not after.
    if std::fs::create_dir_all(sandbox.root().join(DIR)).is_err() {
        return;
    }
    let dir = match sandbox.resolve(DIR, true) {
        Ok(d) => d,
        Err(_) => return, // outside the jail — never happens, never allowed
    };

    let backup = if resolved.exists() {
        // A flat, collision-free name: the relative path with separators
        // replaced, prefixed by its position in the log so repeated edits to
        // one file each keep their own snapshot.
        let n = all().len();
        let flat: String = rel
            .chars()
            .map(|c| if c == '/' || c == '\\' { '_' } else { c })
            .collect();
        let dest = dir.join(format!("{n:04}_{flat}"));
        if std::fs::copy(&resolved, &dest).is_err() {
            return;
        }
        Some(dest)
    } else {
        None
    };

    if let Ok(mut g) = log().lock() {
        g.push(Checkpoint {
            rel,
            backup,
            tool: tool.to_string(),
        });
    }
}

/// Undo the most recent checkpoint. Returns what it did.
pub fn undo(sandbox: &Sandbox) -> Result<String, String> {
    let cp = {
        let mut g = log().lock().map_err(|_| "checkpoint log poisoned")?;
        g.pop().ok_or("nothing to undo")?
    };
    let target = sandbox.resolve(&cp.rel, false)?;
    match &cp.backup {
        Some(b) => {
            std::fs::copy(b, &target).map_err(|e| format!("restore failed: {e}"))?;
            Ok(format!("restored {}", cp.rel))
        }
        None => {
            // The file did not exist before the agent made it.
            let _ = std::fs::remove_file(&target);
            Ok(format!("removed {} (it was newly created)", cp.rel))
        }
    }
}

/// A unified-ish diff of every checkpointed file against what is on disk now.
pub fn diff(sandbox: &Sandbox) -> String {
    let cps = all();
    if cps.is_empty() {
        return "no changes this session".to_string();
    }
    let mut out = String::new();
    for cp in &cps {
        let now = sandbox
            .resolve(&cp.rel, false)
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok());
        let before = cp
            .backup
            .as_ref()
            .and_then(|b| std::fs::read_to_string(b).ok());
        out.push_str(&format!("--- {} ({})\n", cp.rel, cp.tool));
        match (before, now) {
            (None, Some(after)) => {
                for line in after.lines().take(40) {
                    out.push_str(&format!("+ {line}\n"));
                }
            }
            (Some(b), None) => {
                for line in b.lines().take(40) {
                    out.push_str(&format!("- {line}\n"));
                }
            }
            (Some(b), Some(a)) => out.push_str(&line_diff(&b, &a)),
            (None, None) => out.push_str("(gone)\n"),
        }
    }
    out
}

/// A minimal line diff: lines only in `before` are `-`, only in `after` are `+`.
/// Not Myers — enough to show what changed without pulling in a diff crate.
fn line_diff(before: &str, after: &str) -> String {
    let b: Vec<&str> = before.lines().collect();
    let a: Vec<&str> = after.lines().collect();
    let mut out = String::new();
    let mut shown = 0;
    for line in &b {
        if !a.contains(line) {
            out.push_str(&format!("- {line}\n"));
            shown += 1;
        }
        if shown >= 40 {
            break;
        }
    }
    for line in &a {
        if !b.contains(line) {
            out.push_str(&format!("+ {line}\n"));
            shown += 1;
        }
        if shown >= 80 {
            break;
        }
    }
    if out.is_empty() {
        out.push_str("(no textual change)\n");
    }
    out
}

/// `3 change(s): src/a.rs, src/b.rs`
pub fn summary() -> String {
    let cps = all();
    if cps.is_empty() {
        return "no checkpoints this session".to_string();
    }
    let names: Vec<String> = cps.iter().map(|c| c.rel.clone()).collect();
    format!("{} change(s): {}", cps.len(), names.join(", "))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::Duration;

    /// The log is process-wide, like the plan and the MCP registry.
    pub(crate) fn cp_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn sb(root: &Path) -> Sandbox {
        Sandbox::new(root, false, Duration::from_secs(5)).unwrap()
    }

    #[test]
    fn undo_restores_a_file_byte_identical() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        let original = "line one\nline two\nline three\n";
        std::fs::write(&f, original).unwrap();

        take(&sandbox, &f, "edit_file");
        std::fs::write(&f, "totally different\n").unwrap();
        assert_ne!(std::fs::read_to_string(&f).unwrap(), original);

        undo(&sandbox).unwrap();
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            original,
            "undo must restore byte-identical content"
        );
        clear();
    }

    #[test]
    fn undo_removes_a_file_the_agent_created() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("new.txt");

        take(&sandbox, &f, "write_file"); // does not exist yet
        std::fs::write(&f, "created by the agent").unwrap();
        assert!(f.exists());

        undo(&sandbox).unwrap();
        assert!(!f.exists(), "a newly created file should be removed");
        clear();
    }

    #[test]
    fn undo_walks_back_one_change_at_a_time() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        std::fs::write(&f, "v1").unwrap();

        take(&sandbox, &f, "edit_file");
        std::fs::write(&f, "v2").unwrap();
        take(&sandbox, &f, "edit_file");
        std::fs::write(&f, "v3").unwrap();

        undo(&sandbox).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "v2");
        undo(&sandbox).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "v1");
        assert!(undo(&sandbox).is_err(), "nothing left to undo");
        clear();
    }

    #[test]
    fn diff_shows_the_actual_on_disk_delta() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        std::fs::write(&f, "keep\nremove me\n").unwrap();

        take(&sandbox, &f, "edit_file");
        std::fs::write(&f, "keep\nadded line\n").unwrap();

        let out = diff(&sandbox);
        assert!(out.contains("- remove me"), "{out}");
        assert!(out.contains("+ added line"), "{out}");
        assert!(
            !out.contains("- keep"),
            "unchanged lines must not show: {out}"
        );
        assert!(summary().contains("1 change(s)"));
        clear();
    }

    /// The hook lives at the execution site, so an edit made by the model is
    /// checkpointed whether or not the model cooperated.
    #[test]
    fn write_and_edit_through_the_tool_path_are_checkpointed() {
        use super::super::tools::{validate, ToolCall};
        use serde_json::json;
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());

        let call = |name: &str, args| ToolCall {
            name: name.to_string(),
            args,
        };

        // Create, then edit, both through validate -> execute.
        validate(
            &call("write_file", json!({"path":"a.txt","content":"first\n"})),
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);
        validate(
            &call(
                "edit_file",
                json!({"path":"a.txt","old":"first","new":"second"}),
            ),
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);

        assert_eq!(all().len(), 2, "both mutations should be checkpointed");
        assert_eq!(
            std::fs::read_to_string(d.path().join("a.txt")).unwrap(),
            "second\n"
        );

        undo(&sandbox).unwrap();
        assert_eq!(
            std::fs::read_to_string(d.path().join("a.txt")).unwrap(),
            "first\n"
        );
        undo(&sandbox).unwrap();
        assert!(!d.path().join("a.txt").exists());
        clear();
    }

    /// Reads must not create checkpoints — only mutations do.
    #[test]
    fn read_only_tools_take_no_checkpoint() {
        use super::super::tools::{validate, ToolCall};
        use serde_json::json;
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.txt"), "x").unwrap();
        let sandbox = sb(d.path());
        validate(
            &ToolCall {
                name: "read_file".into(),
                args: json!({"path":"a.txt"}),
            },
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);
        assert!(all().is_empty());
        clear();
    }

    /// The store is state inside the workspace, so it obeys the same jail as
    /// every other path. A checkpoint that could be written or restored outside
    /// the sandbox root would be a path-traversal hole.
    #[test]
    fn the_store_stays_inside_the_jail() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());

        let victim = outside.path().join("secret.txt");
        std::fs::write(&victim, "not yours").unwrap();

        // Snapshotting a path outside the root records nothing.
        take(&sandbox, &victim, "write_file");
        assert!(all().is_empty(), "snapshotted a file outside the workspace");

        // And every snapshot that IS taken lands under the workspace.
        let f = d.path().join("a.txt");
        std::fs::write(&f, "x").unwrap();
        take(&sandbox, &f, "write_file");
        for cp in all() {
            if let Some(b) = &cp.backup {
                let canon = std::fs::canonicalize(b).unwrap();
                assert!(
                    canon.starts_with(std::fs::canonicalize(d.path()).unwrap()),
                    "backup escaped the workspace: {}",
                    canon.display()
                );
            }
        }
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "not yours");
        clear();
    }
}
