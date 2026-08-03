//! repack-ghost: restructure a GGUF model into a `.cghost` layer-streaming container.
//!
//! Ghost (layer-streaming) mode executes a model one transformer block at a time, holding
//! only a tiny working window in RAM. GGUF scatters a block's tensors across the file; this
//! tool writes every block's tensors contiguously (one sequential read per block) at the
//! SOURCE quantization — v1 is a pure re-layout, so the streamed path can be parity-gated
//! byte-for-byte against the resident path.

use std::path::PathBuf;

use clap::Parser;

use camelid::gguf::read_metadata;
use camelid::ghost::{
    ensure_distinct_artifact_paths, write_cghost, write_cghost_moe, CghostLayout,
};
use camelid::ghost_hot::write_moe_hot_shadow;
use camelid::model::{Gemma4Binding, LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;

#[derive(Parser)]
#[command(
    name = "repack-ghost",
    about = "Repack a GGUF into a layer-contiguous .cghost streaming container"
)]
struct Args {
    /// Source GGUF model
    model: PathBuf,
    /// Output path (default: <model>.cghost)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Pipeline shard: only write these transformer blocks (e.g. "0..20" or "20..40"),
    /// plus the embedding group when starting at 0 and the output group when ending at the
    /// last layer. Default: the whole model.
    #[arg(long)]
    layers: Option<String>,

    /// Also create a sparse, offset-compatible GGUF containing only the
    /// always-hot non-expert tensors. Use this local-fast-drive shadow as the
    /// model path together with the matching `.cghost`; routed expert tensor
    /// extents are sparse holes apart from tiny identity islands and are not
    /// valid standalone.
    #[arg(long, value_name = "PATH")]
    hot_shadow: Option<PathBuf>,

    /// Create only --hot-shadow and skip the `.cghost` repack. Useful when the
    /// v2 expert artifact already exists. Requires --hot-shadow.
    #[arg(long, requires = "hot_shadow", conflicts_with = "out")]
    hot_shadow_only: bool,
}

fn parse_layers_range(layers: &str) -> anyhow::Result<std::ops::Range<usize>> {
    let parts: Vec<&str> = layers.split("..").collect();
    anyhow::ensure!(parts.len() == 2, "invalid layers range format: {layers}");
    Ok(parts[0].parse::<usize>()?..parts[1].parse::<usize>()?)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let out = args.out.unwrap_or_else(|| {
        let mut p = args.model.clone();
        p.set_extension("cghost");
        p
    });

    // Validate the complete source/output set before the hot shadow is created.
    // In particular, --out and --hot-shadow must never alias one another: the
    // second conversion would otherwise truncate the artifact just completed.
    let mut artifacts = vec![("source GGUF", args.model.as_path())];
    if !args.hot_shadow_only {
        artifacts.push((".cghost output", out.as_path()));
    }
    if let Some(shadow) = args.hot_shadow.as_ref() {
        artifacts.push(("hot-shadow output", shadow.as_path()));
    }
    ensure_distinct_artifact_paths(&artifacts)?;

    println!(
        "[repack-ghost] reading GGUF metadata from {:?}...",
        args.model
    );
    let gguf = read_metadata(&args.model)?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let store = TensorStore::open(&args.model, &gguf);
    let source = args
        .model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let layer_range = args.layers.as_deref().map(parse_layers_range).transpose()?;
    if let Some(shadow) = args.hot_shadow.as_ref() {
        anyhow::ensure!(
            config.gemma4.is_some() && config.moe.is_some(),
            "--hot-shadow requires a full Gemma 4 MoE GGUF"
        );
        println!(
            "[repack-ghost] writing sparse common-core shadow {:?} -> {:?}",
            args.model, shadow
        );
        let report = write_moe_hot_shadow(&args.model, shadow, &gguf, layer_range.clone())?;
        let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "[repack-ghost] hot shadow done: logical {:.2} GiB, allocated {:.2} GiB, copied {:.2} GiB, sparse experts {:.2} GiB ({} extents)",
            gib(report.logical_bytes),
            gib(report.allocated_bytes),
            gib(report.copied_bytes),
            gib(report.sparse_expert_bytes),
            report.expert_extents,
        );
        println!(
            "[repack-ghost] IMPORTANT: {:?} must be used with the matching fingerprinted v2 .cghost; its routed-expert ranges are intentionally sparse except for bounded identity islands",
            shadow
        );
    }

    if args.hot_shadow_only {
        return Ok(());
    }

    let index = if config.gemma4.is_some() && config.moe.is_some() {
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        println!(
            "[repack-ghost] splicing MoE experts for blocks {:?} of {} -> {:?}",
            layer_range.clone().unwrap_or(0..binding.layers.len()),
            binding.layers.len(),
            out
        );
        write_cghost_moe(&store, &binding, &config, &source, &out, layer_range)?
    } else {
        let binding = LlamaTensorBinding::bind(&gguf, &config)?;
        println!(
            "[repack-ghost] repacking dense blocks {:?} of {} -> {:?}",
            layer_range.clone().unwrap_or(0..binding.layers.len()),
            binding.layers.len(),
            out
        );
        write_cghost(&store, &binding, &source, &out, layer_range)?
    };

    let total: u64 = index.groups.iter().map(|g| g.span().1).sum();
    let max_layer = index
        .groups
        .iter()
        .filter(|g| g.id.starts_with("blk."))
        .map(|g| g.span().1)
        .max()
        .unwrap_or(0);
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!(
        "[repack-ghost] done: {:?}, {} groups, {:.2} GiB payload, largest group {:.1} MiB, \
         tied_output={}",
        index.layout,
        index.groups.len(),
        gib(total),
        mib(max_layer),
        index.tied_output
    );
    if index.layout == CghostLayout::MoeExperts {
        println!(
            "[repack-ghost] MoE layout: {} experts/layer, top-{} routed; each cache miss reads one expert group",
            index.expert_count.unwrap_or(0),
            index.expert_used_count.unwrap_or(0)
        );
    }
    Ok(())
}
