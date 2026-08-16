"""vLLM-style block-table KV memory model + COW + hash-chained prefix cache.

Kwon et al. 2023. In July 2026 vLLM deleted the named PagedAttention *CUDA kernel*
(PR #47361) but kept this memory model: fixed-size physical blocks, per-sequence
block tables, and gather/attend over non-contiguous K/V. This file implements that
model — not a CUDA kernel and not a vLLM/FAISS wrapper.

Prefix reuse follows Automatic Prefix Caching: only *full* blocks are indexed, each
keyed by sha256(parent_hash || block_tokens).
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np


class KVCacheOOM(MemoryError):
    """Raised when the block pool has no free physical blocks."""


def block_hash(parent: bytes | None, tokens: tuple[int, ...]) -> bytes:
    """Content-addressed block key: sha256(parent || token ids). Parent is None for block 0."""
    h = hashlib.sha256()
    if parent is not None:
        h.update(parent)
    else:
        h.update(b"\x00ROOT\x00")
    h.update(b"\x00")
    for t in tokens:
        h.update(int(t).to_bytes(8, "little", signed=True))
    return h.digest()


class BlockAllocator:
    """Free list of fixed-size physical KV blocks. No external fragmentation."""

    def __init__(self, num_blocks: int, block_size: int, n_heads: int, head_dim: int, dtype=np.float32):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k = np.zeros((num_blocks, block_size, n_heads, head_dim), dtype=dtype)
        self.v = np.zeros_like(self.k)
        self.refcount = np.zeros(num_blocks, dtype=np.int32)
        self.free: list[int] = list(range(num_blocks))

    def allocate(self) -> int:
        if not self.free:
            raise KVCacheOOM("KV cache OOM: no free blocks")
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def retain(self, block: int) -> int:
        self.refcount[block] += 1
        return block

    def release(self, block: int) -> None:
        self.refcount[block] -= 1
        if self.refcount[block] < 0:
            raise RuntimeError(f"double-free of block {block}")
        if self.refcount[block] == 0:
            self.k[block] = 0
            self.v[block] = 0
            self.free.append(block)

    def cow(self, block: int) -> int:
        """Named paths: exclusive_write (refcount==1) vs clone_shared (refcount>1)."""
        if self.refcount[block] == 1:
            return block  # exclusive_write
        self.refcount[block] -= 1
        nb = self.allocate()
        self.k[nb] = self.k[block].copy()
        self.v[nb] = self.v[block].copy()
        return nb  # clone_shared

    @property
    def free_count(self) -> int:
        return len(self.free)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size


class Sequence:
    """Logical token stream → block table (logical block index → physical block)."""

    def __init__(self, allocator: BlockAllocator):
        self.alloc = allocator
        self.block_table: list[int] = []
        self.seq_len = 0
        self.tokens: list[int] = []

    def append_kv(self, k_t: np.ndarray, v_t: np.ndarray, token: int) -> None:
        """Append one token's K/V. Allocates on demand; COW if the open block is shared."""
        k_t = np.asarray(k_t)
        v_t = np.asarray(v_t)
        bs = self.alloc.block_size
        if self.seq_len % bs == 0:
            self.block_table.append(self.alloc.allocate())
        else:
            last = self.block_table[-1]
            if self.alloc.refcount[last] > 1:
                self.block_table[-1] = self.alloc.cow(last)
        phys = self.block_table[self.seq_len // bs]
        off = self.seq_len % bs
        self.alloc.k[phys, off] = k_t
        self.alloc.v[phys, off] = v_t
        self.seq_len += 1
        self.tokens.append(int(token))

    def fork(self) -> "Sequence":
        """Share physical blocks (beam / branching). Refcounts increment; later writes COW."""
        child = Sequence(self.alloc)
        child.block_table = list(self.block_table)
        child.seq_len = self.seq_len
        child.tokens = list(self.tokens)
        for b in child.block_table:
            self.alloc.retain(b)
        return child

    def free(self) -> None:
        for b in self.block_table:
            self.alloc.release(b)
        self.block_table.clear()
        self.seq_len = 0
        self.tokens.clear()

    def gather_kv(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize K,V in logical order via the block table (physical ids need not be contiguous)."""
        bs = self.alloc.block_size
        k = np.empty((self.seq_len, self.alloc.n_heads, self.alloc.head_dim), dtype=self.alloc.k.dtype)
        v = np.empty_like(k)
        for i in range(self.seq_len):
            phys = self.block_table[i // bs]
            off = i % bs
            k[i] = self.alloc.k[phys, off]
            v[i] = self.alloc.v[phys, off]
        return k, v


def contiguous_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q: (H, D); k,v: (T, H, D). Per-head scaled-dot-product attention."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = np.einsum("hd,thd->ht", q, k) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return np.einsum("ht,thd->hd", p, v)


def paged_attention(q: np.ndarray, seq: Sequence) -> np.ndarray:
    """Named path gather_then_attend: materialize logical K/V, then contiguous attention."""
    k, v = seq.gather_kv()
    return contiguous_attention(q, k, v)


def paged_attention_blockwise(q: np.ndarray, seq: Sequence) -> np.ndarray:
    """Named path blockwise_online: walk the block table with online softmax — no full gather.

    Same math as gather_then_attend; proves attention over non-contiguous physical blocks.
    """
    if seq.seq_len == 0:
        raise ValueError("empty sequence")
    q64 = np.asarray(q, dtype=np.float64)
    scale = 1.0 / math.sqrt(q64.shape[-1])
    h, d = q64.shape
    m = np.full(h, -np.inf, dtype=np.float64)
    l = np.zeros(h, dtype=np.float64)
    out = np.zeros((h, d), dtype=np.float64)
    bs = seq.alloc.block_size
    for i in range(seq.seq_len):
        phys = seq.block_table[i // bs]
        off = i % bs
        k_i = seq.alloc.k[phys, off].astype(np.float64)
        v_i = seq.alloc.v[phys, off].astype(np.float64)
        score = np.einsum("hd,hd->h", q64, k_i) * scale
        m_new = np.maximum(m, score)
        alpha = np.exp(m - m_new)
        beta = np.exp(score - m_new)
        out = out * alpha[:, None] + beta[:, None] * v_i
        l = l * alpha + beta
        m = m_new
    return out / l[:, None]


class PrefixCache:
    """Hash-chained automatic prefix cache (vLLM APC): full blocks only.

    Each full physical block is indexed by sha256(parent_hash || block_tokens).
    Lookup walks the chain until the first miss. Partial trailing blocks are never cached.
    """

    def __init__(self) -> None:
        # OrderedDict: oldest insertion/touch at front for LRU eviction.
        self._by_hash: OrderedDict[bytes, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self._by_hash)

    def insert(self, seq: Sequence) -> int:
        """Cache every *full* block of seq. Returns how many new hashes were inserted.

        Named paths per full block:
          - hash_miss_insert: retain physical block and map hash → id
          - hash_hit_keep: hash already mapped; leave existing physical id (no remap)
        """
        if seq.seq_len != len(seq.tokens):
            raise ValueError("seq.tokens must cover seq_len for prefix hashing")
        bs = seq.alloc.block_size
        n_full = seq.seq_len // bs
        parent: bytes | None = None
        n_new = 0
        for i in range(n_full):
            chunk = tuple(seq.tokens[i * bs : (i + 1) * bs])
            h = block_hash(parent, chunk)
            phys = seq.block_table[i]
            if h not in self._by_hash:
                seq.alloc.retain(phys)
                self._by_hash[h] = phys
                n_new += 1
            else:
                self._by_hash.move_to_end(h)
            parent = h
        return n_new

    def lookup(self, tokens: list[int], alloc: BlockAllocator) -> tuple[Sequence, int]:
        """Longest full-block prefix hit. Hit length is always a multiple of block_size.

        Named paths:
          - cache_hit: retain mapped physical block, append to block table
          - cache_miss: stop walking; return what was hit (possibly empty)
        """
        seq = Sequence(alloc)
        bs = alloc.block_size
        parent: bytes | None = None
        hit_tokens = 0
        n_full = len(tokens) // bs
        for i in range(n_full):
            chunk = tuple(tokens[i * bs : (i + 1) * bs])
            h = block_hash(parent, chunk)
            phys = self._by_hash.get(h)
            if phys is None:
                break  # cache_miss from this block onward
            self._by_hash.move_to_end(h)
            alloc.retain(phys)
            seq.block_table.append(phys)
            hit_tokens += bs
            parent = h
        seq.seq_len = hit_tokens
        seq.tokens = list(tokens[:hit_tokens])
        return seq, hit_tokens

    def evict_lru(self, alloc: BlockAllocator, n: int = 1) -> int:
        """Drop up to n oldest hash entries and release the cache's retain on each."""
        dropped = 0
        while dropped < n and self._by_hash:
            _h, phys = self._by_hash.popitem(last=False)
            alloc.release(phys)
            dropped += 1
        return dropped

    def clear(self, alloc: BlockAllocator) -> None:
        for phys in self._by_hash.values():
            alloc.release(phys)
        self._by_hash.clear()
