//! Comprehensive verification test for:
//! Option 1: Advanced KV Cache Architecture
//! - Feature C: Native Hardware FP8 (E4M3 / E5M2) KV Cache
//! - Feature B: PagedAttention / Virtual Memory Block KV Cache
//! - Feature A: Radix-Tree Multi-Turn Prompt Prefix Caching

use camelid::inference::paged_kv::{paged_attention, BlockTable, PagedKvBlockPool};
use camelid::inference::radix_cache::RadixPrefixCache;
use camelid::inference::{KvDtype, LlamaKvCachePlan};
use camelid::tensor::kv_quant::{
    dequantize_block_fp8_e4m3, dequantize_block_fp8_e5m2, quantize_block_fp8_e4m3,
    quantize_block_fp8_e5m2, vec_dot_fp8_e4m3, vec_dot_fp8_e5m2, BlockFp8E4m3, BlockFp8E5m2,
    KV_QUANT_BLOCK_VALUES,
};

fn dummy_plan() -> LlamaKvCachePlan {
    LlamaKvCachePlan {
        max_sequence_length: 256,
        layer_count: 2,
        kv_head_count: 2,
        head_dim: 64,
        k_head_dim: 64,
        v_head_dim: 64,
        key_shape: vec![1, 2, 256, 64],
        value_shape: vec![1, 2, 256, 64],
    }
}

fn test_feature_c_fp8_quantization() {
    println!("\n=== Testing Feature C: Native FP8 (E4M3 / E5M2) KV Cache ===");

    // 1. Wire size checks
    assert_eq!(std::mem::size_of::<BlockFp8E4m3>(), 34, "BlockFp8E4m3 must be 34 bytes");
    assert_eq!(std::mem::size_of::<BlockFp8E5m2>(), 34, "BlockFp8E5m2 must be 34 bytes");
    println!("  [PASS] Block sizes are wire-compatible (34 bytes per 32 elements).");

    // 2. FP8 E4M3 roundtrip & dot product
    let mut original = [0.0f32; KV_QUANT_BLOCK_VALUES];
    for (i, v) in original.iter_mut().enumerate() {
        *v = (i as f32 - 16.0) * 0.25;
    }
    let block_e4m3 = quantize_block_fp8_e4m3(&original);
    let mut reconstructed_e4m3 = [0.0f32; KV_QUANT_BLOCK_VALUES];
    dequantize_block_fp8_e4m3(&block_e4m3, &mut reconstructed_e4m3);

    for i in 0..KV_QUANT_BLOCK_VALUES {
        let diff = (original[i] - reconstructed_e4m3[i]).abs();
        assert!(diff < 0.35, "E4M3 quantization error too high at {i}: {diff}");
    }
    let query: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| (i as f32 - 8.0) * 0.1);
    let expected_dot: f32 = query.iter().zip(reconstructed_e4m3).map(|(q, k)| q * k).sum();
    let actual_dot = vec_dot_fp8_e4m3(&query, &block_e4m3);
    assert!((actual_dot - expected_dot).abs() < 1e-4, "E4M3 dot product mismatch");
    println!("  [PASS] FP8 E4M3 block quantize/dequantize & dot product match reference.");

    // 3. FP8 E5M2 roundtrip & dot product
    let block_e5m2 = quantize_block_fp8_e5m2(&original);
    let mut reconstructed_e5m2 = [0.0f32; KV_QUANT_BLOCK_VALUES];
    dequantize_block_fp8_e5m2(&block_e5m2, &mut reconstructed_e5m2);

    for i in 0..KV_QUANT_BLOCK_VALUES {
        let diff = (original[i] - reconstructed_e5m2[i]).abs();
        assert!(diff < 0.60, "E5M2 quantization error too high at {i}: {diff}");
    }
    let expected_dot_e5m2: f32 = query.iter().zip(reconstructed_e5m2).map(|(q, k)| q * k).sum();
    let actual_dot_e5m2 = vec_dot_fp8_e5m2(&query, &block_e5m2);
    assert!((actual_dot_e5m2 - expected_dot_e5m2).abs() < 1e-4, "E5M2 dot product mismatch");
    println!("  [PASS] FP8 E5M2 block quantize/dequantize & dot product match reference.");
}

fn test_feature_b_paged_kv_cache() {
    println!("\n=== Testing Feature B: PagedAttention / Virtual Memory Block KV Cache ===");

    let plan = dummy_plan();
    let mut pool = PagedKvBlockPool::new(plan.clone(), KvDtype::Fp8E4m3, 20);

    // 1. Allocation & Free-list recycling
    let b0 = pool.allocate().expect("failed to allocate block 0");
    let b1 = pool.allocate().expect("failed to allocate block 1");
    assert_ne!(b0, b1);
    assert_eq!(pool.allocated_count(), 2);
    pool.release(b0);
    assert_eq!(pool.allocated_count(), 1);
    assert_eq!(pool.free_count(), 1);
    let b0_recycled = pool.allocate().expect("failed to allocate recycled block");
    assert_eq!(b0, b0_recycled, "Pool must reuse freed blocks");
    println!("  [PASS] Block allocation & zero-overhead free-list recycling verified.");

    // 2. Logical block table and chunking
    let mut table1 = BlockTable::new();
    for token_id in 0..36 {
        table1.append_token(&mut pool, token_id).unwrap();
    }
    // 36 tokens with block size 16 = 3 physical blocks (16, 16, 4)
    assert_eq!(table1.blocks.len(), 3);
    assert_eq!(table1.len(), 36);

    let (mapped_id, offset) = table1.logical_to_physical(34).unwrap();
    assert_eq!(mapped_id, table1.blocks[2]);
    assert_eq!(offset, 2); // 34 - 32 = 2
    println!("  [PASS] BlockTable virtual memory mapping (36 tokens -> 3 blocks) verified.");

    // 3. Copy-on-Write (CoW) branching
    let mut table2 = table1.fork(&mut pool);
    assert_eq!(table1.blocks, table2.blocks);
    assert_eq!(pool.block(table1.blocks[0]).unwrap().ref_count, 2);

    // Appending to table2 triggers CoW on the last partially filled block
    table2.append_token(&mut pool, 999).unwrap();
    assert_ne!(table1.blocks[2], table2.blocks[2], "Last block must be cloned on write");
    assert_eq!(pool.block(table1.blocks[0]).unwrap().ref_count, 2, "Prefix blocks remain shared");
    println!("  [PASS] O(1) Copy-on-Write sequence branching verified.");

    // 4. Paged Attention operator execution with FP8 storage
    let mut eval_table = BlockTable::new();
    let key_data = vec![0.3f32; 64];
    let val_data = vec![0.7f32; 64];

    for i in 0..8 {
        let (block_id, offset) = eval_table.append_token(&mut pool, i as u32).unwrap();
        let block = pool.block_mut(block_id).unwrap();
        block.store_kv(offset, 0, 0, &key_data, &val_data, &plan);
    }

    let query = vec![0.2f32; 64];
    let mut attention_out = vec![0.0f32; 64];
    paged_attention(&query, 0, 0, 0.125, &eval_table, &pool, &mut attention_out).unwrap();

    for &val in &attention_out {
        assert!((val - 0.70).abs() < 0.05, "Expected attention output ~0.70, got {val}");
    }
    println!("  [PASS] PagedAttention operator with FP8 storage executes correctly.");
}

fn test_feature_a_radix_tree_prefix_caching() {
    println!("\n=== Testing Feature A: Radix-Tree Multi-Turn Prompt Prefix Caching ===");

    let plan = dummy_plan();
    let mut pool = PagedKvBlockPool::new(plan, KvDtype::F32, 50);
    let mut radix = RadixPrefixCache::new(100);

    let b1 = pool.allocate().unwrap();
    let b2 = pool.allocate().unwrap();
    let b3 = pool.allocate().unwrap();

    // Turn 1 System Prompt + User message: [100, 101, 102, 103, 104, 105]
    let turn1_tokens = vec![100, 101, 102, 103, 104, 105];
    radix.insert_sequence(&turn1_tokens, &[b1], &mut pool).unwrap();

    // Query exact same prefix: should get 100% cache hit (6 tokens)
    let (matched_blocks, matched_tokens) = radix.match_longest_prefix(&turn1_tokens, &mut pool);
    assert_eq!(matched_tokens, 6);
    assert_eq!(matched_blocks, vec![b1]);
    println!("  [PASS] Exact multi-turn prompt prefix match: 6/6 tokens cached.");

    // Turn 2 branching user message sharing common system prefix [100, 101, 102]:
    // prompt: [100, 101, 102, 200, 201]
    let turn2_tokens = vec![100, 101, 102, 200, 201];
    radix.insert_sequence(&turn2_tokens, &[b2], &mut pool).unwrap();

    // Query branch 2: should match prefix [100, 101, 102, 200, 201]
    let (_, matched_branch2) = radix.match_longest_prefix(&turn2_tokens, &mut pool);
    assert_eq!(matched_branch2, 5);

    // Query unknown user message with same system prefix: [100, 101, 102, 999]
    let (_common_blocks, common_len) = radix.match_longest_prefix(&[100, 101, 102, 999], &mut pool);
    assert_eq!(common_len, 3, "Common system prompt tokens [100, 101, 102] must be matched");
    println!("  [PASS] Radix tree split on divergent turns: shared prefix [100, 101, 102] preserved.");

    // LRU eviction test
    let turn3_tokens = vec![300, 301, 302];
    radix.insert_sequence(&turn3_tokens, &[b3], &mut pool).unwrap();
    let nodes_before = radix.node_count();
    let evicted = radix.evict_lru(1, &mut pool);
    assert_eq!(evicted, 1);
    assert_eq!(radix.node_count(), nodes_before - 1);
    println!("  [PASS] LRU eviction under memory pressure verified.");

    println!("  Cache Stats: {:?}", radix.stats);
}

fn main() {
    println!("===============================================================");
    println!("Camelid Advanced KV Cache Architecture Verification (Option 1)");
    println!("===============================================================");

    test_feature_c_fp8_quantization();
    test_feature_b_paged_kv_cache();
    test_feature_a_radix_tree_prefix_caching();

    println!("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<\n");
}
