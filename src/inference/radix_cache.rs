//! Radix-Tree Multi-Turn Prompt Prefix Caching.
//!
//! Replaces single-slot or naive prefix caching with a dynamic prefix tree (compact trie)
//! of token sequences backed by shared physical KV blocks from `PagedKvBlockPool`.
//!
//! Multi-turn chat sessions and shared system prompts frequently share long common prefixes
//! (e.g. system instructions, tool definitions, earlier conversation turns). By indexing
//! KV cache blocks in a Radix Tree:
//! 1. Prompt prefill can match the longest existing prefix and immediately resume from it
//!    without recomputing any attention or feed-forward operations.
//! 2. Shared physical blocks are reference-counted, saving significant memory.
//! 3. LRU eviction guarantees bounded cache size under memory pressure.

use std::collections::HashMap;
use std::time::Instant;

use crate::inference::paged_kv::{PagedKvBlockPool, PhysicalBlockId, KV_BLOCK_TOKENS};
use crate::Result;

/// A node in the Radix Tree representing a sequence of tokens and associated physical blocks.
#[derive(Debug, Clone)]
pub struct RadixNode {
    pub id: usize,
    pub parent: Option<usize>,
    /// Child transitions keyed by the FIRST token of the child's edge sequence.
    pub children: HashMap<u32, usize>,
    /// Token sequence on this edge.
    pub tokens: Vec<u32>,
    /// Physical blocks holding the KV activations for these tokens.
    pub blocks: Vec<PhysicalBlockId>,
    /// Last access timestamp for LRU ordering.
    pub last_accessed: Instant,
    /// Reference count of active inference sessions currently referencing this node.
    pub ref_count: usize,
}

impl RadixNode {
    pub fn new(id: usize, parent: Option<usize>, tokens: Vec<u32>, blocks: Vec<PhysicalBlockId>) -> Self {
        Self {
            id,
            parent,
            children: HashMap::new(),
            tokens,
            blocks,
            last_accessed: Instant::now(),
            ref_count: 0,
        }
    }
}

/// Statistics for prefix cache performance monitoring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RadixCacheStats {
    pub queries: u64,
    pub hits: u64,
    pub tokens_queried: u64,
    pub tokens_saved: u64,
    pub evicted_nodes: u64,
}

/// Dynamic Radix-Tree Prefix Cache for multi-turn conversations.
#[derive(Debug)]
pub struct RadixPrefixCache {
    nodes: Vec<Option<RadixNode>>,
    root_id: usize,
    free_node_ids: Vec<usize>,
    pub max_nodes: usize,
    pub stats: RadixCacheStats,
}

impl RadixPrefixCache {
    pub fn new(max_nodes: usize) -> Self {
        let mut cache = Self {
            nodes: Vec::new(),
            root_id: 0,
            free_node_ids: Vec::new(),
            max_nodes,
            stats: RadixCacheStats::default(),
        };

        // Initialize root node representing empty prefix
        let root = RadixNode::new(0, None, Vec::new(), Vec::new());
        cache.nodes.push(Some(root));
        cache
    }

    fn alloc_node_id(&mut self) -> usize {
        if let Some(id) = self.free_node_ids.pop() {
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(None);
            id
        }
    }

    /// Match longest token prefix existing in the tree.
    /// Returns `(matched_blocks, matched_token_count)`.
    pub fn match_longest_prefix(
        &mut self,
        tokens: &[u32],
        pool: &mut PagedKvBlockPool,
    ) -> (Vec<PhysicalBlockId>, usize) {
        self.stats.queries += 1;
        self.stats.tokens_queried += tokens.len() as u64;

        let mut current_id = self.root_id;
        let mut token_idx = 0;
        let mut matched_blocks = Vec::new();
        let now = Instant::now();

        while token_idx < tokens.len() {
            let next_token = tokens[token_idx];
            let child_id = match self.nodes[current_id].as_ref().and_then(|n| n.children.get(&next_token).copied()) {
                Some(id) => id,
                None => break,
            };

            let child = self.nodes[child_id].as_mut().unwrap();
            child.last_accessed = now;

            // Check how many tokens match along this edge
            let edge_tokens = &child.tokens;
            let mut edge_matched = 0;
            while edge_matched < edge_tokens.len() && (token_idx + edge_matched) < tokens.len() {
                if edge_tokens[edge_matched] == tokens[token_idx + edge_matched] {
                    edge_matched += 1;
                } else {
                    break;
                }
            }

            if edge_matched == edge_tokens.len() {
                // Entire edge matched, retain and collect blocks
                for &b in &child.blocks {
                    pool.retain(b);
                    matched_blocks.push(b);
                }
                token_idx += edge_matched;
                current_id = child_id;
            } else {
                // Partial edge match: can only reuse complete blocks up to edge_matched
                let usable_blocks = edge_matched / KV_BLOCK_TOKENS;
                for &b in &child.blocks[..usable_blocks] {
                    pool.retain(b);
                    matched_blocks.push(b);
                }
                token_idx += usable_blocks * KV_BLOCK_TOKENS;
                break;
            }
        }

        if token_idx > 0 {
            self.stats.hits += 1;
            self.stats.tokens_saved += token_idx as u64;
        }

        (matched_blocks, token_idx)
    }

    /// Insert sequence of tokens and their associated physical blocks into the Radix Tree.
    pub fn insert_sequence(
        &mut self,
        tokens: &[u32],
        blocks: &[PhysicalBlockId],
        pool: &mut PagedKvBlockPool,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        let mut current_id = self.root_id;
        let mut token_offset = 0;
        let mut block_offset = 0;
        let now = Instant::now();

        while token_offset < tokens.len() {
            let next_token = tokens[token_offset];
            let child_id_opt = self.nodes[current_id].as_ref().and_then(|n| n.children.get(&next_token).copied());

            match child_id_opt {
                Some(child_id) => {
                    let (common_len, edge_len) = {
                        let child = self.nodes[child_id].as_ref().unwrap();
                        let mut common = 0;
                        while common < child.tokens.len()
                            && (token_offset + common) < tokens.len()
                            && child.tokens[common] == tokens[token_offset + common]
                        {
                            common += 1;
                        }
                        (common, child.tokens.len())
                    };

                    if common_len == edge_len {
                        // Full match along this edge: advance down
                        let blocks_in_edge = self.nodes[child_id].as_ref().unwrap().blocks.len();
                        token_offset += edge_len;
                        block_offset += blocks_in_edge;
                        current_id = child_id;
                        self.nodes[current_id].as_mut().unwrap().last_accessed = now;
                    } else {
                        // Divergence on edge: SPLIT child into split_node and remaining child
                        let split_id = self.alloc_node_id();
                        let split_blocks_count = common_len.div_ceil(KV_BLOCK_TOKENS).min(
                            self.nodes[child_id].as_ref().unwrap().blocks.len()
                        );

                        // Extract child properties
                        let (child_tokens, child_blocks, _child_children) = {
                            let child = self.nodes[child_id].as_mut().unwrap();
                            child.parent = Some(split_id);
                            let remaining_tokens = child.tokens.split_off(common_len);
                            let remaining_blocks = if split_blocks_count < child.blocks.len() {
                                child.blocks.split_off(split_blocks_count)
                            } else {
                                Vec::new()
                            };
                            let prefix_tokens = std::mem::replace(&mut child.tokens, remaining_tokens);
                            let prefix_blocks = std::mem::replace(&mut child.blocks, remaining_blocks);
                            (prefix_tokens, prefix_blocks, std::mem::take(&mut child.children))
                        };

                        let first_remaining_token = self.nodes[child_id].as_ref().unwrap().tokens[0];

                        // Create split node
                        let mut split_node = RadixNode::new(split_id, Some(current_id), child_tokens, child_blocks);
                        split_node.children.insert(first_remaining_token, child_id);
                        self.nodes[split_id] = Some(split_node);

                        // Attach split node to current parent
                        self.nodes[current_id].as_mut().unwrap().children.insert(next_token, split_id);

                        token_offset += common_len;
                        block_offset += split_blocks_count;

                        // Now add the new branch for the rest of tokens
                        if token_offset < tokens.len() {
                            let new_leaf_id = self.alloc_node_id();
                            let remaining_new_tokens = tokens[token_offset..].to_vec();
                            let remaining_new_blocks = blocks[block_offset..].to_vec();
                            for &b in &remaining_new_blocks {
                                pool.retain(b);
                            }

                            let new_leaf = RadixNode::new(
                                new_leaf_id,
                                Some(split_id),
                                remaining_new_tokens,
                                remaining_new_blocks,
                            );
                            self.nodes[new_leaf_id] = Some(new_leaf);
                            self.nodes[split_id].as_mut().unwrap().children.insert(tokens[token_offset], new_leaf_id);
                        }
                        return Ok(());
                    }
                }
                None => {
                    // No child matching next_token: create a new leaf node directly under current_id
                    let new_leaf_id = self.alloc_node_id();
                    let remaining_tokens = tokens[token_offset..].to_vec();
                    let remaining_blocks = blocks[block_offset..].to_vec();
                    for &b in &remaining_blocks {
                        pool.retain(b);
                    }

                    let new_leaf = RadixNode::new(
                        new_leaf_id,
                        Some(current_id),
                        remaining_tokens,
                        remaining_blocks,
                    );
                    self.nodes[new_leaf_id] = Some(new_leaf);
                    self.nodes[current_id].as_mut().unwrap().children.insert(next_token, new_leaf_id);
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Evict oldest unreferenced leaf nodes under memory pressure.
    pub fn evict_lru(&mut self, count: usize, pool: &mut PagedKvBlockPool) -> usize {
        let mut evicted = 0;

        for _ in 0..count {
            // Find leaf node with ref_count == 0 and oldest last_accessed
            let mut oldest_leaf: Option<(usize, Instant)> = None;

            for (idx, node_opt) in self.nodes.iter().enumerate() {
                if idx == self.root_id {
                    continue;
                }
                if let Some(node) = node_opt {
                    if node.children.is_empty() && node.ref_count == 0 {
                        match oldest_leaf {
                            None => oldest_leaf = Some((idx, node.last_accessed)),
                            Some((_, oldest_time)) if node.last_accessed < oldest_time => {
                                oldest_leaf = Some((idx, node.last_accessed));
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some((leaf_id, _)) = oldest_leaf {
                let leaf = self.nodes[leaf_id].take().unwrap();
                for &b in &leaf.blocks {
                    pool.release(b);
                }

                if let Some(parent_id) = leaf.parent {
                    if let Some(parent) = self.nodes[parent_id].as_mut() {
                        parent.children.retain(|_, &mut child_id| child_id != leaf_id);
                    }
                }

                self.free_node_ids.push(leaf_id);
                evicted += 1;
                self.stats.evicted_nodes += 1;
            } else {
                break;
            }
        }

        evicted
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::KvDtype;

    fn dummy_plan() -> crate::inference::kv_cache::LlamaKvCachePlan {
        crate::inference::kv_cache::LlamaKvCachePlan {
            max_sequence_length: 128,
            layer_count: 1,
            kv_head_count: 1,
            head_dim: 64,
            k_head_dim: 64,
            v_head_dim: 64,
            key_shape: vec![1, 1, 128, 64],
            value_shape: vec![1, 1, 128, 64],
        }
    }

    #[test]
    fn test_radix_prefix_insert_and_match() {
        let plan = dummy_plan();
        let mut pool = PagedKvBlockPool::new(plan, KvDtype::F32, 10);
        let mut cache = RadixPrefixCache::new(50);

        let b0 = pool.allocate().unwrap();
        let b1 = pool.allocate().unwrap();

        // Insert prompt 1: [1, 2, 3, 4, 5]
        let tokens1 = vec![1, 2, 3, 4, 5];
        cache.insert_sequence(&tokens1, &[b0], &mut pool).unwrap();

        // Query identical prompt: should match all 5 tokens
        let (matched_blocks, matched_len) = cache.match_longest_prefix(&tokens1, &mut pool);
        assert_eq!(matched_len, 5);
        assert_eq!(matched_blocks, vec![b0]);

        // Insert diverging prompt: [1, 2, 3, 9, 10]
        let tokens2 = vec![1, 2, 3, 9, 10];
        cache.insert_sequence(&tokens2, &[b1], &mut pool).unwrap();

        // Query common prefix [1, 2, 3, 44]: should match [1, 2, 3]
        let (_, matched_common) = cache.match_longest_prefix(&[1, 2, 3, 44], &mut pool);
        assert_eq!(matched_common, 3);
    }

    #[test]
    fn test_radix_lru_eviction() {
        let plan = dummy_plan();
        let mut pool = PagedKvBlockPool::new(plan, KvDtype::F32, 10);
        let mut cache = RadixPrefixCache::new(50);

        let b0 = pool.allocate().unwrap();
        cache.insert_sequence(&[10, 20, 30], &[b0], &mut pool).unwrap();
        assert_eq!(cache.node_count(), 2); // root + 1 node

        let evicted = cache.evict_lru(1, &mut pool);
        assert_eq!(evicted, 1);
        assert_eq!(cache.node_count(), 1); // only root remains
    }
}
