"""vLLM-style paged KV: block table + attention over non-contiguous physical blocks.

Kwon et al. 2023. vLLM later deleted the named PagedAttention CUDA kernel
(PR #47361, 2026) but kept this block-table memory model. This file is the
memory model plus gather-then-attend — not a CUDA kernel.
"""

from __future__ import annotations

import math

import numpy as np


class KVCacheOOM(MemoryError):
    """Raised when the block pool is exhausted."""


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
        """Copy-on-write: if shared, clone into a new physical block."""
        if self.refcount[block] == 1:
            return block
        self.refcount[block] -= 1
        nb = self.allocate()
        self.k[nb] = self.k[block].copy()
        self.v[nb] = self.v[block].copy()
        return nb

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

    def append_kv(self, k_t: np.ndarray, v_t: np.ndarray, token: int | None = None) -> None:
        """Append one token's K/V. Allocates a block on demand; COW if the slot is shared."""
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
        if token is not None:
            self.tokens.append(int(token))

    def fork(self) -> "Sequence":
        """Share physical blocks (beam / prefix). Refcounts increment; writes COW."""
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
        """Materialize K,V in logical order via the block table (blocks need not be contiguous)."""
        bs = self.alloc.block_size
        k = np.empty((self.seq_len, self.alloc.n_heads, self.alloc.head_dim), dtype=self.alloc.k.dtype)
        v = np.empty_like(k)
        for i in range(self.seq_len):
            phys = self.block_table[i // bs]
            off = i % bs
            k[i] = self.alloc.k[phys, off]
            v[i] = self.alloc.v[phys, off]
        return k, v


def paged_attention(q: np.ndarray, seq: Sequence) -> np.ndarray:
    """Attention of Q against paged K,V. Q is (n_heads, head_dim) — the new query."""
    k, v = seq.gather_kv()
    return contiguous_attention(q, k, v)


def contiguous_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q: (H, D); k,v: (T, H, D). Per-head softmax attention."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scale = 1.0 / math.sqrt(q.shape[-1])
    # scores (H, T)
    scores = np.einsum("hd,thd->ht", q, k) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return np.einsum("ht,thd->hd", p, v)


class PrefixCache:
    """Exact-prefix cache: token tuple → retained block table snapshot.

    A hit reuses the same physical block IDs for the shared prefix. The last
    block is shared with COW so a later write diverges without mutating siblings.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[int, ...], tuple[list[int], int]] = {}

    def insert(self, seq: Sequence) -> None:
        if not seq.tokens:
            return
        key = tuple(seq.tokens)
        # Retain so the cache owns a ref even after the inserting seq is freed.
        for b in seq.block_table:
            seq.alloc.retain(b)
        prev = self._store.get(key)
        if prev is not None:
            for b in prev[0]:
                seq.alloc.release(b)
        self._store[key] = (list(seq.block_table), seq.seq_len)

    def lookup(self, tokens: list[int], alloc: BlockAllocator) -> tuple[Sequence, int]:
        """Longest common prefix against any stored sequence. Reuses those physical blocks."""
        best_n = 0
        best_table: list[int] | None = None
        for key, (table, _slen) in self._store.items():
            n = 0
            while n < len(key) and n < len(tokens) and key[n] == tokens[n]:
                n += 1
            if n > best_n:
                best_n = n
                best_table = table
        seq = Sequence(alloc)
        if best_n == 0 or best_table is None:
            return seq, 0
        n_blocks = (best_n + alloc.block_size - 1) // alloc.block_size
        seq.block_table = list(best_table[:n_blocks])
        seq.seq_len = best_n
        seq.tokens = list(tokens[:best_n])
        for b in seq.block_table:
            alloc.retain(b)
        return seq, best_n

    def clear(self, alloc: BlockAllocator) -> None:
        for table, _ in self._store.values():
            for b in table:
                alloc.release(b)
        self._store.clear()
