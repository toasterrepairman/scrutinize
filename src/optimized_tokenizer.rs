use std::collections::HashMap;

/// Optimized trie-based tokenizer for fast longest-match tokenization.
/// Uses a prefix tree (trie) structure to achieve O(m) lookup time where m is the
/// length of the text, instead of O(n*m*k) where k is vocabulary size.

#[derive(Debug)]
pub struct TokenizerTrie {
    root: TrieNode,
    max_token_len: usize,
}

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<u8, Box<TrieNode>>,
    token_id: Option<usize>,
    token_data: Option<TokenMetadata>,
}

#[derive(Debug, Clone)]
struct TokenMetadata {
    score: f32,
    token_type: u32,
}

impl TokenizerTrie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::default(),
            max_token_len: 0,
        }
    }

    /// Build the trie from a list of tokens
    pub fn build(&mut self, tokens: &[(usize, String, f32, u32)]) {
        self.max_token_len = 0;

        for (id, token, score, token_type) in tokens {
            self.max_token_len = self.max_token_len.max(token.len());
            self.insert(token.as_bytes(), *id, *score, *token_type);
        }
    }

    fn insert(&mut self, token_bytes: &[u8], id: usize, score: f32, token_type: u32) {
        let mut node = &mut self.root;

        for &byte in token_bytes {
            node = node.children
                .entry(byte)
                .or_insert_with(|| Box::new(TrieNode::default()));
        }

        node.token_id = Some(id);
        node.token_data = Some(TokenMetadata { score, token_type });
    }

    /// Find the longest matching token starting at the given position in the text.
    /// Returns (token_id, token_length, score, token_type) or None if no match.
    #[inline]
    pub fn find_longest_match(&self, text: &[u8], start: usize) -> Option<(usize, usize, f32, u32)> {
        let mut node = &self.root;
        let mut last_match: Option<(usize, usize, f32, u32)> = None;
        let mut current_len = 0;

        // Limit search to max token length for better cache behavior
        let max_search = (text.len() - start).min(self.max_token_len);

        for i in 0..max_search {
            let byte = text[start + i];

            if let Some(child) = node.children.get(&byte) {
                node = child;
                current_len += 1;

                // If this node represents a complete token, record it
                if let (Some(id), Some(metadata)) = (node.token_id, &node.token_data) {
                    last_match = Some((id, current_len, metadata.score, metadata.token_type));
                }
            } else {
                // No more matches possible
                break;
            }
        }

        last_match
    }

    /// Tokenize text using greedy longest-match algorithm with trie optimization.
    /// Returns vector of (token_id, token_length, score, token_type).
    pub fn tokenize(&self, text: &str) -> Vec<(usize, usize, f32, u32)> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < bytes.len() {
            if let Some((id, len, score, token_type)) = self.find_longest_match(bytes, pos) {
                tokens.push((id, len, score, token_type));
                pos += len;
            } else {
                // No match found, skip one byte
                pos += 1;
            }
        }

        tokens
    }

    /// Optimized tokenization with early termination for UI display.
    /// Stops after max_tokens to avoid processing unnecessarily long inputs.
    pub fn tokenize_limited(&self, text: &str, max_tokens: usize) -> (Vec<(usize, usize, f32, u32)>, bool) {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut pos = 0;
        let mut truncated = false;

        while pos < bytes.len() && tokens.len() < max_tokens {
            if let Some((id, len, score, token_type)) = self.find_longest_match(bytes, pos) {
                tokens.push((id, len, score, token_type));
                pos += len;
            } else {
                // No match found, skip one byte
                pos += 1;
            }
        }

        if pos < bytes.len() {
            truncated = true;
        }

        (tokens, truncated)
    }

    /// Get memory usage statistics for the trie
    pub fn memory_stats(&self) -> TrieStats {
        let mut stats = TrieStats::default();
        self.count_nodes(&self.root, &mut stats);
        stats
    }

    fn count_nodes(&self, node: &TrieNode, stats: &mut TrieStats) {
        stats.node_count += 1;

        if node.token_id.is_some() {
            stats.token_count += 1;
        }

        stats.edge_count += node.children.len();

        for child in node.children.values() {
            self.count_nodes(child, stats);
        }
    }
}

#[derive(Debug, Default)]
pub struct TrieStats {
    pub node_count: usize,
    pub token_count: usize,
    pub edge_count: usize,
}

impl TrieStats {
    pub fn estimated_memory_bytes(&self) -> usize {
        // Rough estimate:
        // - Each node: HashMap overhead + Option fields ≈ 128 bytes
        // - Each edge: 1 byte key + pointer ≈ 16 bytes
        self.node_count * 128 + self.edge_count * 16
    }
}

/// SIMD-accelerated helper functions for common patterns
pub mod simd {
    /// Find the next potential token boundary (whitespace or punctuation).
    /// This can be used to skip ahead in certain tokenization scenarios.
    #[inline]
    pub fn find_next_boundary(text: &[u8], start: usize) -> Option<usize> {
        text[start..].iter()
            .position(|&b| matches!(b, b' ' | b'\n' | b'\t' | b'\r' | b'.' | b',' | b'!' | b'?'))
            .map(|pos| start + pos)
    }

    /// Fast check if a sequence contains only ASCII characters.
    /// Modern CPUs can do this very efficiently.
    #[inline]
    pub fn is_ascii(text: &[u8]) -> bool {
        text.iter().all(|&b| b < 128)
    }

    /// Count ASCII whitespace characters using SIMD-friendly code.
    /// Compiler should auto-vectorize this on supported platforms.
    #[inline]
    pub fn count_whitespace(text: &[u8]) -> usize {
        text.iter()
            .filter(|&&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let mut trie = TokenizerTrie::new();

        let tokens = vec![
            (0, "hello".to_string(), 0.0, 0),
            (1, " ".to_string(), 0.0, 0),
            (2, "world".to_string(), 0.0, 0),
            (3, "hel".to_string(), 0.0, 0), // Shorter token should not be preferred
        ];

        trie.build(&tokens);

        let result = trie.tokenize("hello world");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 0); // "hello"
        assert_eq!(result[1].0, 1); // " "
        assert_eq!(result[2].0, 2); // "world"
    }

    #[test]
    fn test_longest_match() {
        let mut trie = TokenizerTrie::new();

        let tokens = vec![
            (0, "a".to_string(), 0.0, 0),
            (1, "ab".to_string(), 0.0, 0),
            (2, "abc".to_string(), 0.0, 0),
        ];

        trie.build(&tokens);

        let result = trie.tokenize("abc");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 2); // Should match "abc", not "ab" or "a"
    }

    #[test]
    fn test_no_match() {
        let mut trie = TokenizerTrie::new();

        let tokens = vec![
            (0, "hello".to_string(), 0.0, 0),
        ];

        trie.build(&tokens);

        let result = trie.tokenize("xyz");
        assert_eq!(result.len(), 0); // No matches
    }

    #[test]
    fn test_limited_tokenization() {
        let mut trie = TokenizerTrie::new();

        let tokens = vec![
            (0, "a".to_string(), 0.0, 0),
        ];

        trie.build(&tokens);

        let (result, truncated) = trie.tokenize_limited("aaaaaaaaaa", 5);
        assert_eq!(result.len(), 5);
        assert!(truncated);
    }

    #[test]
    fn test_utf8_tokens() {
        let mut trie = TokenizerTrie::new();

        let tokens = vec![
            (0, "你好".to_string(), 0.0, 0),
            (1, "世界".to_string(), 0.0, 0),
        ];

        trie.build(&tokens);

        let result = trie.tokenize("你好世界");
        assert_eq!(result.len(), 2);
    }
}
