import assert from 'node:assert/strict'
import { groupByModelFamily, modelFamily } from '../src/lib/modelFamilies.js'

assert.equal(modelFamily({ filename: 'Llama-3.2-3B-Instruct-Q8_0.gguf', architecture: 'llama' }), 'Llama')
assert.equal(modelFamily({ name: 'Bonsai 27B Q1_0', architecture: 'qwen35' }), 'Bonsai')
assert.equal(modelFamily({ name: 'Ornith 4B Instruct', architecture: 'qwen35' }), 'Ornith')
assert.equal(modelFamily({ name: 'DeepSeek R1 Distill Qwen 7B', architecture: 'qwen2' }), 'DeepSeek')
assert.equal(modelFamily({ name: 'DeepSeek R1 0528 Qwen3 8B', architecture: 'qwen3' }), 'DeepSeek')
assert.equal(modelFamily({ name: 'Mistral Nemo Instruct 2407', architecture: 'llama' }), 'Mistral')
assert.equal(modelFamily({ name: 'Qwen3 14B', architecture: 'qwen3' }), 'Qwen')
assert.equal(modelFamily({ name: 'LFM2.5 2.6B', architecture: 'lfm2' }), 'LFM')
assert.equal(modelFamily({ name: 'SmolLM2-1.7B-Instruct', architecture: 'llama' }), 'SmolLM')
assert.equal(modelFamily({ repo_id: 'HuggingFaceTB/SmolLM3-3B', architecture: 'smollm3' }), 'SmolLM')
assert.equal(modelFamily({ architecture: 'qwen3moe' }), 'Qwen')
assert.equal(modelFamily({ architecture: 'brand-new-arch' }), 'brand-new-arch')
assert.equal(modelFamily({}), 'Other')

const rows = [
  { name: 'Llama 3.2 1B', architecture: 'llama' },
  { name: 'Qwen3 4B', architecture: 'qwen3' },
  { name: 'Llama 3.1 8B', architecture: 'llama' },
  { name: 'Bonsai 4B', architecture: 'qwen3' },
]
const groups = groupByModelFamily(rows)
assert.deepEqual(groups.map((group) => group.family), ['Llama', 'Qwen', 'Bonsai'])
assert.equal(groups[0].items.length, 2)
assert.equal(groups[0].items[0], rows[0], 'grouping must preserve row identity and catalog order')

console.log('model family smoke: ok')
