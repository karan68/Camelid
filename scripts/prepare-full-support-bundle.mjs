#!/usr/bin/env node
import { createHash } from 'node:crypto'
import os from 'node:os'
import { execFileSync } from 'node:child_process'
import { chmod, mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const repoRoot = resolve(args.get('repo-root') || '.')
const utcStamp = args.get('utc') || isoStamp(new Date())
const gitHead = git(['rev-parse', 'HEAD'], repoRoot)
const gitHeadShort = git(['rev-parse', '--short=12', 'HEAD'], repoRoot)
const originMain = git(['rev-parse', 'origin/main'], repoRoot)
const branch = git(['branch', '--show-current'], repoRoot)
const outDir = resolve(args.get('out-dir') || join(repoRoot, 'target', `full-support-${utcStamp}-head-${gitHeadShort}`))
const outDirRelative = relative(repoRoot, outDir) || '.'
const qaBundleRoot = 'qa/evidence-bundles/four-row-public-20260503T024327Z'
const perfEnvelopePath = 'qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json'
const validationNotePath = 'qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md'
const toolchainCommand = repoCommand('./scripts/with-rustup-cargo.sh +1.87.0 build --release --bin backendinference')
const apiBase = '${CAMELID_API_BASE:-http://127.0.0.1:8181}'
const frontendUrl = '${CAMELID_FRONTEND_URL:-http://127.0.0.1:4175}'
const llamaBase = '${LLAMA3_LLAMA_SERVER_URL:-http://127.0.0.1:8183}'
const tinyLlamaBase = '${TINYLLAMA_LLAMA_SERVER_URL:-http://127.0.0.1:8183}'
const llamaServerBin = '${CAMELID_LLAMA_SERVER_BIN:-target/reference/llama.cpp/build/bin/llama-server}'
const llamaTokenizeBin = '${CAMELID_LLAMA_TOKENIZE_BIN:-target/reference/llama.cpp/build/bin/llama-tokenize}'
const modelDir = '${CAMELID_MODEL_DIR:?set CAMELID_MODEL_DIR to the GGUF directory}'

const rows = [
  {
    row_id: 'tinyllama_1_1b_chat_q8_0',
    display_name: 'TinyLlama 1.1B Chat Q8_0',
    public_status: 'supported_current_gate',
    model_file: 'tinyllama-1.1b-chat-v1.0.Q8_0.gguf',
    model_id: 'tinyllama-q8',
    compatibility_row: 'tinyllama_1_1b_chat_q8_0',
    expected_compatibility_status: 'supported_current_gate',
    expect_contract_supported: true,
    expect_webui_chat: 'enabled',
    expected_model_sha256: 'a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59',
    template_family: 'tinyllama_marker',
    carry_forward_bundle: `${qaBundleRoot}/tinyllama_1_1b_chat_q8_0.bundle.json`,
    notes: [
      'Current public support is already a real TinyLlama gate, but this row still needs the same durable current-head bundle shape as the three Llama rows.',
      'The Llama-3-specific template/context packs do not apply unchanged here; keep TinyLlama evidence exact-row and marker-template scoped.'
    ],
    blockers: [
      'Fresh current-head API/WebUI/perf artifacts are still needed in a durable target/full-support root.',
      'Do not imply support for adjacent TinyLlama quantizations or other families.'
    ],
    tracks: [
      {
        id: 'compact-parity',
        kind: 'parity',
        status: 'ready_to_run',
        description: 'Refresh bounded TinyLlama hello parity on current head.',
        pack_path: 'qa/prompt-packs/tinyllama-hello-5tok.json',
        command: repoCommand(`node scripts/chat-parity-tinyllama.mjs --backend ${apiBase} --llama-url ${tinyLlamaBase} --model \"${modelDir}/tinyllama-1.1b-chat-v1.0.Q8_0.gguf\" --model-id tinyllama-q8 --llama-server \"${llamaServerBin}\" --start-llama-server --message hello --max-tokens 5 --require-generated-match --diagnostics-out ROW_ROOT/parity-compact/hello-5tok.json`)
      },
      {
        id: 'broader-parity',
        kind: 'parity',
        status: 'carry_forward_only',
        description: 'Preserve the existing five-prompt/50-token TinyLlama gate while a fresh current-head rerun is scheduled.',
        carry_forward_artifacts: [
          'target/edge-prompt-audit-fixed-20260428T1530/short.json',
          'target/edge-prompt-audit-fixed-20260428T1530/trailing-spaces.json',
          'target/edge-prompt-audit-fixed-20260428T1530/special-chars.json',
          'target/edge-prompt-audit-fixed-20260428T1530/longer.json',
          'target/edge-prompt-dequant-default-20260428T1604/multiline-long-default-50.json'
        ],
        command: repoCommand('python3 - <<\'PY\'\nimport json, os, pathlib\npaths = [\n  "target/edge-prompt-audit-fixed-20260428T1530/short.json",\n  "target/edge-prompt-audit-fixed-20260428T1530/trailing-spaces.json",\n  "target/edge-prompt-audit-fixed-20260428T1530/special-chars.json",\n  "target/edge-prompt-audit-fixed-20260428T1530/longer.json",\n  "target/edge-prompt-dequant-default-20260428T1604/multiline-long-default-50.json",\n]\nreport = {"checked": []}\nfor path in paths:\n  data = json.loads(pathlib.Path(path).read_text())\n  report["checked"].append({\n    "path": path,\n    "prompt_tokens_match": data.get("prompt_tokens_match"),\n    "generated_text_match": data.get("generated_text_match"),\n    "backend_tokens": len(data.get("backend_generated_tokens", [])),\n    "llama_tokens": len(data.get("llama_generated_tokens", data.get("llama_generated_tokens_from_text", []))),\n  })\nout_path = pathlib.Path(os.environ["ROW_ROOT"]) / "broader-parity" / "carry-forward-summary.json"\nout_path.write_text(json.dumps(report, indent=2) + "\\n")\nprint("wrote", out_path)\nPY')
      },
      {
        id: 'chat-template-shapes',
        kind: 'template',
        status: 'ready_to_run',
        description: 'Run the exact-row TinyLlama marker-template shape pack.',
        pack_path: 'qa/prompt-packs/tinyllama-chat-template-shapes.json',
        command: repoCommand(`node scripts/run-llama3-prompt-pack.mjs --backend ${apiBase} --llama-url ${tinyLlamaBase} --model "${modelDir}/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" --model-id tinyllama-q8 --llama-server "${llamaServerBin}" --llama-tokenize "${llamaTokenizeBin}" --start-llama-server --pack qa/prompt-packs/tinyllama-chat-template-shapes.json --out-dir ROW_ROOT/chat-template-shapes --wait-ms 180000 --require-prompt-match --require-generated-match`)
      },
      {
        id: 'context-512',
        kind: 'context',
        status: 'ready_to_run',
        description: 'Run the bounded TinyLlama 512-context pack and preserve success or failure durably.',
        pack_path: 'qa/prompt-packs/tinyllama-context-512-smoke.json',
        command: repoCommand(`node scripts/run-llama3-prompt-pack.mjs --backend ${apiBase} --llama-url ${tinyLlamaBase} --model "${modelDir}/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" --model-id tinyllama-q8 --llama-server "${llamaServerBin}" --llama-tokenize "${llamaTokenizeBin}" --start-llama-server --pack qa/prompt-packs/tinyllama-context-512-smoke.json --out-dir ROW_ROOT/context-512 --wait-ms 180000 --require-prompt-match --require-generated-match`)
      },
      {
        id: 'api-webui-smoke',
        kind: 'api_webui',
        status: 'ready_to_run',
        description: 'Refresh current-head TinyLlama load/completions/chat/frontend smoke.',
        command: repoCommand(`node scripts/model-promotion-smoke-bundle.mjs --api ${apiBase} --frontend ${frontendUrl} --model \"${modelDir}/tinyllama-1.1b-chat-v1.0.Q8_0.gguf\" --model-id tinyllama-q8 --out-dir ROW_ROOT/api-webui --message hello --max-tokens 1 --temperature 0 --expect-compatibility-row tinyllama_1_1b_chat_q8_0 --expect-compatibility-status supported_current_gate --expect-contract-supported true --expect-webui-chat enabled`)
      },
      {
        id: 'perf-rss-portability',
        kind: 'perf',
        status: 'ready_to_run',
        description: 'Capture host facts plus RSS after load/1tok/5tok/API-WebUI smoke.',
        command: perfCommand('tinyllama-1.1b-chat-v1.0.Q8_0.gguf', 'tinyllama-q8')
      }
    ]
  },
  {
    row_id: 'llama32_1b_instruct_q8_0',
    display_name: 'Llama 3.2 1B Instruct Q8_0',
    public_status: 'supported_exact_row_smoke',
    model_file: 'Llama-3.2-1B-Instruct-Q8_0.gguf',
    model_id: 'llama32-1b-q8',
    compatibility_row: 'llama32_1b_instruct_q8_0',
    expected_compatibility_status: 'supported_exact_row_smoke',
    expect_contract_supported: true,
    expect_webui_chat: 'enabled',
    expected_model_sha256: '432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3',
    template_family: 'llama3_instruct',
    carry_forward_bundle: `${qaBundleRoot}/llama32_1b_instruct_q8_0.bundle.json`,
    notes: [
      'Broader prompt-pack evidence exists, but the public claim remains exact-row short-chat smoke.',
      'Promotion-grade longer-context, broader template coverage, and portability still need Ubuntu current-head reruns.'
    ],
    blockers: [
      'No durable current-head target/full-support evidence root exists yet for compact/broader/template/512/API-WebUI/perf together.',
      'Do not imply neighboring Llama 3.2 rows or other quantizations are supported.'
    ],
    tracks: llamaTracks({
      modelFile: 'Llama-3.2-1B-Instruct-Q8_0.gguf',
      modelId: 'llama32-1b-q8',
      compatibilityRow: 'llama32_1b_instruct_q8_0',
      compatibilityStatus: 'supported_exact_row_smoke',
      expectContractSupported: true,
      expectWebUiChat: 'enabled',
      broaderPack: 'qa/prompt-packs/llama3-broader-repro-3prompt.json',
      contextWaitMs: 180000,
      perfWaitMs: 180000,
    })
  },
  {
    row_id: 'llama32_3b_instruct_q8_0',
    display_name: 'Llama 3.2 3B Instruct Q8_0',
    public_status: 'supported_exact_row_smoke',
    model_file: 'Llama-3.2-3B-Instruct-Q8_0.gguf',
    model_id: 'llama32-3b-q8',
    compatibility_row: 'llama32_3b_instruct_q8_0',
    expected_compatibility_status: 'supported_exact_row_smoke',
    expect_contract_supported: true,
    expect_webui_chat: 'enabled',
    expected_model_sha256: 'b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921',
    template_family: 'llama3_instruct',
    carry_forward_bundle: `${qaBundleRoot}/llama32_3b_instruct_q8_0.bundle.json`,
    notes: [
      'The post-Q8-dot broader three-prompt pack passed for prompt tokens, generated token IDs, and generated text.',
      'Longer context, broader template behavior, and stronger portability/perf evidence remain the release blocker.'
    ],
    blockers: [
      'Current public support is still exact-row smoke only.',
      'Do not broaden beyond the exact 3B Instruct Q8_0 row without fresh Ubuntu artifacts and synchronized docs/API/frontend changes.'
    ],
    tracks: llamaTracks({
      modelFile: 'Llama-3.2-3B-Instruct-Q8_0.gguf',
      modelId: 'llama32-3b-q8',
      compatibilityRow: 'llama32_3b_instruct_q8_0',
      compatibilityStatus: 'supported_exact_row_smoke',
      expectContractSupported: true,
      expectWebUiChat: 'enabled',
      broaderPack: 'qa/prompt-packs/llama3-broader-repro-3prompt.json',
      contextWaitMs: 300000,
      perfWaitMs: 300000,
    })
  },
  {
    row_id: 'llama3_8b_instruct_q8_0',
    display_name: 'Llama 3 8B Instruct Q8_0',
    public_status: 'supported_exact_row_smoke',
    model_file: 'Meta-Llama-3-8B-Instruct-Q8_0.gguf',
    model_id: 'llama3-8b-q8',
    compatibility_row: 'llama3_8b_instruct_gguf',
    expected_compatibility_status: 'supported_exact_row_smoke',
    expect_contract_supported: true,
    expect_webui_chat: 'enabled',
    expected_model_sha256: '583c616da14b82930f887f991ab446711da0b029166200b67892d7c9f8f45958',
    template_family: 'llama3_instruct',
    carry_forward_bundle: `${qaBundleRoot}/llama3_8b_instruct_q8_0.bundle.json`,
    notes: [
      'Short smoke is supported for the exact row only; the broader 5-token Ubuntu pack passed on the tracked GGUF.',
      'The first bounded 512-context pack timed out at /v1/chat/completions after 300000 ms on current head and remains the blocker.'
    ],
    blockers: [
      '512-context parity is still blocked on Ubuntu current head; keep that failure preserved side-by-side with passing short smoke.',
      'Do not broaden to neighboring Llama sizes, quantizations, longer contexts, or other template families.'
    ],
    tracks: llamaTracks({
      modelFile: 'Meta-Llama-3-8B-Instruct-Q8_0.gguf',
      modelId: 'llama3-8b-q8',
      compatibilityRow: 'llama3_8b_instruct_gguf',
      compatibilityStatus: 'supported_exact_row_smoke',
      expectContractSupported: true,
      expectWebUiChat: 'enabled',
      broaderPack: 'qa/prompt-packs/llama3-broader-repro-3prompt.json',
      contextWaitMs: 300000,
      perfWaitMs: 1200000,
      contextTrackStatus: 'known_blocker',
      contextTrackNotes: [
        'Known blocker from qa/validation-notes/2026-05-03-ubuntu-toolchain-and-8b-context.md: Camelid timed out at POST /v1/chat/completions after 300000 ms while llama.cpp finished the same 512-context reference prompt + 5-token completion.',
        'Keep the failure durable inside the new bundle instead of papering over it.'
      ],
    })
  }
]

await mkdir(outDir, { recursive: true })
await mkdir(join(outDir, 'commands'), { recursive: true })

const manifest = {
  schema: 'camelid.full_support.execution_bundle.v1',
  generated_utc: new Date().toISOString(),
  bundle_root: outDirRelative,
  purpose: 'Current-head full-support execution scaffold plus durable exact-row carry-forward references.',
  git: {
    repo_root: '.',
    branch,
    head: gitHead,
    head_short: gitHeadShort,
    origin_main: originMain,
    dirty_paths: gitLines(['status', '--short'], repoRoot),
  },
  host: {
    hostname: os.hostname(),
    platform: os.platform(),
    release: os.release(),
    arch: os.arch(),
    node: process.version,
  },
  ubuntu_validation_guardrail: 'Use the canonical Ubuntu validation host for promotion-grade Llama runtime evidence. Local Mac work is for docs/recon/light prep only.',
  required_tracks: ['compact-parity', 'broader-parity', 'chat-template-shapes', 'context-512', 'api-webui-smoke', 'perf-rss-portability'],
  prerequisites: {
    build_command: toolchainCommand,
    backend_binary: 'target/release/backendinference',
    reference_llama_server: llamaServerBin,
    reference_llama_tokenize: llamaTokenizeBin,
    required_env: {
      CAMELID_MODEL_DIR: 'Directory containing the exact GGUF rows.',
      CAMELID_API_BASE: 'Camelid API base URL (default http://127.0.0.1:8181).',
      CAMELID_FRONTEND_URL: 'Camelid frontend URL (default http://127.0.0.1:4175).',
      LLAMA3_LLAMA_SERVER_URL: 'Reference llama.cpp server URL for Llama 3 rows (default http://127.0.0.1:8183).',
      TINYLLAMA_LLAMA_SERVER_URL: 'Reference llama.cpp server URL for TinyLlama (default http://127.0.0.1:8183).',
      CAMELID_LLAMA_SERVER_BIN: 'Path to llama.cpp llama-server binary.',
      CAMELID_LLAMA_TOKENIZE_BIN: 'Path to llama.cpp llama-tokenize binary.',
    },
  },
  carry_forward_public_refs: {
    normalized_bundle_root: qaBundleRoot,
    perf_portability_envelope: perfEnvelopePath,
    validation_note: validationNotePath,
  },
  rows: rows.map(row => summarizeRow(outDir, row)),
}

await writeJson(join(outDir, 'manifest.json'), manifest)
await writeFile(join(outDir, 'README.md'), renderReadme(manifest), 'utf8')
await writeExecutable(join(outDir, 'commands', 'build-current-head.sh'), topLevelShellScript(toolchainCommand))
await writeExecutable(join(outDir, 'commands', 'capture-host-facts.sh'), topLevelShellScript(hostFactsCommand()))
await writeExecutable(join(outDir, 'commands', 'run-all-rows.sh'), topLevelShellScript(renderRunAll(rows)))

for (const row of rows) {
  const rowRoot = join(outDir, row.row_id)
  await mkdir(join(rowRoot, 'commands'), { recursive: true })
  await mkdir(join(rowRoot, 'evidence'), { recursive: true })
  const rowManifest = summarizeRow(outDir, row)
  await writeJson(join(rowRoot, 'manifest.json'), rowManifest)
  await writeFile(join(rowRoot, 'README.md'), renderRowReadme(row, rowManifest), 'utf8')
  await writeExecutable(join(rowRoot, 'commands', '00-model-sha256.sh'), rowShellScript(modelShaCommand(row.model_file)))
  for (const [index, track] of row.tracks.entries()) {
    const scriptName = `${String(index + 1).padStart(2, '0')}-${track.id}.sh`
    await writeExecutable(join(rowRoot, 'commands', scriptName), rowShellScript(track.command))
  }
}

await writeSha256Sums(outDir)

console.log(`bundle_root=${outDir}`)
console.log(`manifest=${join(outDir, 'manifest.json')}`)
console.log(`head=${gitHead}`)
console.log(`origin_main=${originMain}`)
console.log(`rows=${rows.length}`)

function summarizeRow(outDir, row) {
  const rowRoot = join(outDir, row.row_id)
  const rowRootRelative = relative(repoRoot, rowRoot) || '.'
  return {
    row_id: row.row_id,
    display_name: row.display_name,
    public_status: row.public_status,
    model_file: row.model_file,
    model_id: row.model_id,
    model_path_env: `${modelDir}/${row.model_file}`,
    expected_model_sha256: row.expected_model_sha256,
    template_family: row.template_family,
    compatibility_row: row.compatibility_row,
    expected_compatibility_status: row.expected_compatibility_status,
    expect_contract_supported: row.expect_contract_supported,
    expect_webui_chat: row.expect_webui_chat,
    row_root: rowRootRelative,
    carry_forward_bundle: row.carry_forward_bundle,
    notes: row.notes,
    blockers: row.blockers,
    tracks: row.tracks.map((track, index) => ({
      index: index + 1,
      id: track.id,
      kind: track.kind,
      status: track.status,
      description: track.description,
      pack_path: track.pack_path ?? null,
      carry_forward_artifacts: track.carry_forward_artifacts ?? [],
      notes: track.notes ?? [],
      command_file: join(rowRootRelative, 'commands', `${String(index + 1).padStart(2, '0')}-${track.id}.sh`),
    })),
  }
}

function llamaTracks({ modelFile, modelId, compatibilityRow, compatibilityStatus, expectContractSupported, expectWebUiChat, broaderPack, contextWaitMs, perfWaitMs, contextTrackStatus = 'ready_to_run', contextTrackNotes = [] }) {
  return [
    {
      id: 'compact-parity',
      kind: 'parity',
      status: 'ready_to_run',
      description: 'Refresh compact-header hello parity at 5 tokens on current head.',
      command: repoCommand(`node scripts/chat-parity-llama3.mjs --backend ${apiBase} --llama-url ${llamaBase} --model \"${modelDir}/${modelFile}\" --model-id ${modelId} --llama-server \"${llamaServerBin}\" --llama-tokenize \"${llamaTokenizeBin}\" --start-llama-server --message hello --max-tokens 5 --render-mode compact --wait-ms ${Math.max(contextWaitMs, 120000)} --require-prompt-match --require-generated-match --diagnostics-out ROW_ROOT/parity-compact/hello-5tok.json`)
    },
    {
      id: 'broader-parity',
      kind: 'parity',
      status: 'ready_to_run',
      description: 'Run the broader three-prompt pack and require prompt/generated parity.',
      pack_path: broaderPack,
      command: repoCommand(`node scripts/run-llama3-prompt-pack.mjs --backend ${apiBase} --llama-url ${llamaBase} --model \"${modelDir}/${modelFile}\" --model-id ${modelId} --llama-server \"${llamaServerBin}\" --llama-tokenize \"${llamaTokenizeBin}\" --start-llama-server --pack ${broaderPack} --out-dir ROW_ROOT/broader-parity --wait-ms ${Math.max(contextWaitMs, 120000)} --require-prompt-match --require-generated-match`)
    },
    {
      id: 'chat-template-shapes',
      kind: 'template',
      status: 'ready_to_run',
      description: 'Run the chat-template-shapes pack to broaden template coverage on the exact row.',
      pack_path: 'qa/prompt-packs/llama3-chat-template-shapes.json',
      command: repoCommand(`node scripts/run-llama3-prompt-pack.mjs --backend ${apiBase} --llama-url ${llamaBase} --model \"${modelDir}/${modelFile}\" --model-id ${modelId} --llama-server \"${llamaServerBin}\" --llama-tokenize \"${llamaTokenizeBin}\" --start-llama-server --pack qa/prompt-packs/llama3-chat-template-shapes.json --out-dir ROW_ROOT/chat-template-shapes --wait-ms ${Math.max(contextWaitMs, 120000)} --require-prompt-match --require-generated-match`)
    },
    {
      id: 'context-512',
      kind: 'context',
      status: contextTrackStatus,
      description: 'Run the bounded 512-context pack and preserve success or failure durably.',
      pack_path: 'qa/prompt-packs/llama3-context-512-smoke.json',
      notes: contextTrackNotes,
      command: repoCommand(`node scripts/run-llama3-prompt-pack.mjs --backend ${apiBase} --llama-url ${llamaBase} --model \"${modelDir}/${modelFile}\" --model-id ${modelId} --llama-server \"${llamaServerBin}\" --llama-tokenize \"${llamaTokenizeBin}\" --start-llama-server --pack qa/prompt-packs/llama3-context-512-smoke.json --out-dir ROW_ROOT/context-512 --wait-ms ${contextWaitMs} --require-prompt-match --require-generated-match`)
    },
    {
      id: 'api-webui-smoke',
      kind: 'api_webui',
      status: 'ready_to_run',
      description: 'Refresh exact-row /api/models/load, /v1/models, /v1/completions, /v1/chat/completions, and frontend smoke.',
      command: repoCommand(`node scripts/model-promotion-smoke-bundle.mjs --api ${apiBase} --frontend ${frontendUrl} --model \"${modelDir}/${modelFile}\" --model-id ${modelId} --out-dir ROW_ROOT/api-webui --message hello --max-tokens 1 --temperature 0 --expect-compatibility-row ${compatibilityRow} --expect-compatibility-status ${compatibilityStatus} --expect-contract-supported ${String(expectContractSupported)} --expect-webui-chat ${expectWebUiChat}`)
    },
    {
      id: 'perf-rss-portability',
      kind: 'perf',
      status: 'ready_to_run',
      description: 'Capture host facts, versions, model SHA, smoke timing, and backend RSS snapshots in one portable note.',
      command: perfCommand(modelFile, modelId, perfWaitMs)
    },
  ]
}

function perfCommand(modelFile, modelId, waitMs = 300000) {
  return [
    'set -euo pipefail',
    'cd "$REPO_ROOT"',
    'mkdir -p "$ROW_ROOT/perf-rss-portability"',
    `MODEL=\"${modelDir}/${modelFile}\"`,
    `MODEL_ID=\"${modelId}\"`,
    `API_BASE=\"${apiBase}\"`,
    `FRONTEND_URL=\"${frontendUrl}\"`,
    `WAIT_MS=\"${waitMs}\"`,
    'date -u +%FT%TZ | tee "$ROW_ROOT/perf-rss-portability/captured-at.txt"',
    'uname -a | tee "$ROW_ROOT/perf-rss-portability/uname.txt"',
    'hostname | tee "$ROW_ROOT/perf-rss-portability/hostname.txt"',
    'node --version | tee "$ROW_ROOT/perf-rss-portability/node-version.txt"',
    './scripts/with-rustup-cargo.sh --version | tee "$ROW_ROOT/perf-rss-portability/cargo-version.txt"',
    'free -h | tee "$ROW_ROOT/perf-rss-portability/free.txt"',
    'df -h / | tee "$ROW_ROOT/perf-rss-portability/disk-root.txt"',
    'shasum -a 256 "$MODEL" | tee "$ROW_ROOT/perf-rss-portability/model.sha256.txt"',
    `node scripts/model-promotion-smoke-bundle.mjs --api ${apiBase} --frontend ${frontendUrl} --model \"${modelDir}/${modelFile}\" --model-id ${modelId} --out-dir \"$ROW_ROOT/perf-rss-portability/api-webui-smoke\" --message hello --max-tokens 1 --temperature 0 || true`,
    "pgrep -f 'target/release/backendinference serve' | tail -n 1 | tee \"$ROW_ROOT/perf-rss-portability/backend.pid.txt\"",
    "if [ -s \"$ROW_ROOT/perf-rss-portability/backend.pid.txt\" ]; then ps -o pid,rss,vsz,etime,command -p \"$(cat \"$ROW_ROOT/perf-rss-portability/backend.pid.txt\")\" | tee \"$ROW_ROOT/perf-rss-portability/backend.ps.txt\"; fi",
  ].join('\n')
}

function modelShaCommand(modelFile) {
  return [
    'set -euo pipefail',
    'cd "$REPO_ROOT"',
    `MODEL=\"${modelDir}/${modelFile}\"`,
    'mkdir -p "$ROW_ROOT/evidence"',
    'shasum -a 256 "$MODEL" | tee "$ROW_ROOT/evidence/model.sha256.txt"',
  ].join('\n')
}

function hostFactsCommand() {
  return [
    'set -euo pipefail',
    'cd "$REPO_ROOT"',
    'date -u +%FT%TZ',
    'git rev-parse HEAD',
    'git status --short',
    'uname -a',
    'hostname',
    'node --version',
    './scripts/with-rustup-cargo.sh --version',
    'free -h',
    'df -h /',
  ].join('\n')
}

function renderRunAll(rows) {
  return [
    'set -euo pipefail',
    'cd "$BUNDLE_ROOT"',
    './commands/build-current-head.sh',
    './commands/capture-host-facts.sh > host-facts.txt',
    ...rows.flatMap(row => [
      `echo "== ${row.row_id} =="`,
      `( cd "$BUNDLE_ROOT/${row.row_id}/commands" && ./00-model-sha256.sh )`,
      ...row.tracks.map((track, index) => `( cd "$BUNDLE_ROOT/${row.row_id}/commands" && ./${String(index + 1).padStart(2, '0')}-${track.id}.sh )`),
    ]),
  ].join('\n')
}

function renderReadme(manifest) {
  return `# Full-support current-head execution bundle\n\nGenerated: ${manifest.generated_utc}\n\nGit head: \`${manifest.git.head}\`\nOrigin/main: \`${manifest.git.origin_main}\`\n\nThis bundle is a durable execution scaffold for the four exact rows Tim cares about. It does **not** widen support by itself. Its job is to normalize the evidence shape so each row has the same folders, command files, model SHA capture, and carry-forward references before or during Ubuntu reruns.\n\nRequired tracks per row:\n- compact parity\n- broader parity\n- chat-template shapes\n- 512-context\n- API/WebUI smoke\n- perf/RSS/portability\n\nTop-level commands:\n- \`commands/build-current-head.sh\`\n- \`commands/capture-host-facts.sh\`\n- \`commands/run-all-rows.sh\`\n\nGuardrails:\n- Use the canonical Ubuntu validation host for promotion-grade Llama runtime evidence.\n- Keep claims exact-row only unless docs, API, frontend, and artifacts all agree.\n- Preserve known blockers durably instead of deleting them, especially the 8B 512-context timeout.\n\nCarry-forward public references:\n- \`${manifest.carry_forward_public_refs.normalized_bundle_root}\`\n- \`${manifest.carry_forward_public_refs.perf_portability_envelope}\`\n- \`${manifest.carry_forward_public_refs.validation_note}\`\n`}

function renderRowReadme(row, manifest) {
  const tracks = manifest.tracks.map(track => `- ${track.id}: ${track.status} — ${track.description}`).join('\n')
  const blockers = row.blockers.map(blocker => `- ${blocker}`).join('\n')
  return `# ${row.display_name}\n\nPublic status: ${row.public_status}\nExpected model SHA256: \`${row.expected_model_sha256}\`\nCarry-forward bundle: \`${row.carry_forward_bundle}\`\n\nTracks:\n${tracks}\n\nBlockers:\n${blockers}\n`}

function repoCommand(command) {
  return `cd "$REPO_ROOT" && ${command}`
}

function shellScript(body) {
  return `#!/usr/bin/env bash\nset -euo pipefail\n\n${body}\n`
}

function topLevelShellScript(body) {
  return shellScript([
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
    'BUNDLE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"',
    'REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"',
    'export BUNDLE_ROOT REPO_ROOT',
    body,
  ].join('\n'))
}

function rowShellScript(body) {
  return shellScript([
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
    'ROW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"',
    'BUNDLE_ROOT="$(cd -- "$ROW_ROOT/.." && pwd)"',
    'REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"',
    'export ROW_ROOT BUNDLE_ROOT REPO_ROOT',
    body.replaceAll('ROW_ROOT', '$ROW_ROOT'),
  ].join('\n'))
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++i] : 'true')
    parsed.set(key, value)
  }
  return parsed
}

function writeJson(path, payload) {
  return writeFile(path, `${JSON.stringify(payload, null, 2)}\n`, 'utf8')
}

async function writeExecutable(path, content) {
  await writeFile(path, content, 'utf8')
  await chmod(path, 0o755)
}

async function writeSha256Sums(rootDir) {
  const files = []
  await collectFiles(rootDir, rootDir, files)
  const lines = []
  for (const file of files.sort()) {
    if (file === 'SHA256SUMS') continue
    const hash = createHash('sha256').update(await readFile(join(rootDir, file))).digest('hex')
    lines.push(`${hash}  ${file}`)
  }
  await writeFile(join(rootDir, 'SHA256SUMS'), `${lines.join('\n')}\n`, 'utf8')
}

async function collectFiles(rootDir, currentDir, output) {
  const entries = await readdir(currentDir, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = join(currentDir, entry.name)
    if (entry.isDirectory()) {
      await collectFiles(rootDir, fullPath, output)
      continue
    }
    const info = await stat(fullPath)
    if (!info.isFile()) continue
    output.push(relative(rootDir, fullPath))
  }
}

function git(args, cwd) {
  return execFileSync('git', args, { cwd, encoding: 'utf8' }).trim()
}

function gitLines(args, cwd) {
  const value = git(args, cwd)
  return value ? value.split(/\r?\n/) : []
}

function isoStamp(date) {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
}
