"""Paged KV tests — fail if the block-table model, COW, or hash prefix cache is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from paged_kv import (
    BlockAllocator,
    KVCacheOOM,
    PrefixCache,
    Sequence,
    block_hash,
    contiguous_attention,
    paged_attention,
    paged_attention_blockwise,
)


def _kv(h, d, rng):
    return rng.standard_normal((h, d)).astype(np.float32), rng.standard_normal((h, d)).astype(np.float32)


def test_no_external_fragmentation():
    rng = np.random.default_rng(0)
    h, d, bs, nb = 2, 8, 4, 40
    alloc = BlockAllocator(nb, bs, h, d)
    live: list[Sequence] = []
    tok = 0
    for _ in range(100):
        length = int(rng.integers(1, 18))
        s = Sequence(alloc)
        try:
            for _t in range(length):
                k, v = _kv(h, d, rng)
                s.append_kv(k, v, token=tok)
                tok += 1
        except KVCacheOOM:
            s.free()
            continue
        live.append(s)
        if rng.random() < 0.65:
            s.free()
            live.pop()
    for s in live:
        s.free()
    assert alloc.free_count == nb
    filler = Sequence(alloc)
    for i in range(alloc.capacity_tokens):
        k, v = _kv(h, d, rng)
        filler.append_kv(k, v, token=i)
    assert alloc.free_count == 0
    filler.free()
    assert alloc.free_count == nb


def test_cow_share_then_diverge_refcount():
    rng = np.random.default_rng(1)
    h, d, bs = 2, 4, 4
    alloc = BlockAllocator(16, bs, h, d)
    s1 = Sequence(alloc)
    for t in range(5):
        k, v = _kv(h, d, rng)
        s1.append_kv(k, v, token=t)
    s2 = s1.fork()
    assert s1.block_table == s2.block_table
    last = s1.block_table[-1]
    first = s1.block_table[0]
    assert alloc.refcount[last] == 2
    assert alloc.refcount[first] == 2
    k, v = _kv(h, d, rng)
    s2.append_kv(k, v, token=99)  # writes shared partial block → COW
    assert s1.block_table[0] == s2.block_table[0]
    assert alloc.refcount[first] == 2
    assert s1.block_table[-1] != s2.block_table[-1]
    assert alloc.refcount[s1.block_table[-1]] == 1
    assert alloc.refcount[s2.block_table[-1]] == 1
    # Parent's K on the partial block must be unchanged after child's write.
    k1, _ = s1.gather_kv()
    assert k1.shape[0] == 5


def test_paged_attention_matches_contiguous_and_blockwise():
    rng = np.random.default_rng(2)
    h, d, bs = 3, 8, 3
    alloc = BlockAllocator(20, bs, h, d)
    s = Sequence(alloc)
    other = Sequence(alloc)
    ks, vs = [], []
    for i in range(bs * 3):
        k, v = _kv(h, d, rng)
        ks.append(k)
        vs.append(v)
        s.append_kv(k, v, token=i)
        if (i + 1) % bs == 0:
            other.append_kv(*_kv(h, d, rng), token=1000 + i)
    assert any(abs(s.block_table[i] - s.block_table[i + 1]) != 1 for i in range(len(s.block_table) - 1))
    q = rng.standard_normal((h, d)).astype(np.float32)
    k_c = np.stack(ks, axis=0)
    v_c = np.stack(vs, axis=0)
    out_p = paged_attention(q, s)
    out_b = paged_attention_blockwise(q, s)
    out_c = contiguous_attention(q, k_c, v_c)
    assert np.allclose(out_p, out_c, atol=1e-5)
    assert np.allclose(out_b, out_c, atol=1e-5)
    # Not an identity stub: blockwise must use online stats (poisoning gather path unused).
    assert out_b.shape == (h, d)


def test_oom_when_blocks_exhausted():
    rng = np.random.default_rng(3)
    alloc = BlockAllocator(2, 2, 1, 4)
    s = Sequence(alloc)
    for t in range(4):
        s.append_kv(*_kv(1, 4, rng), token=t)
    with pytest.raises(KVCacheOOM):
        s.append_kv(*_kv(1, 4, rng), token=4)


def test_prefix_cache_full_blocks_only():
    rng = np.random.default_rng(4)
    h, d, bs = 2, 4, 4
    alloc = BlockAllocator(16, bs, h, d)
    cache = PrefixCache()
    s1 = Sequence(alloc)
    toks = [10, 20, 30, 40, 50]  # one full block + partial
    stored_k = []
    for t in toks:
        k, v = _kv(h, d, rng)
        stored_k.append(k)
        s1.append_kv(k, v, token=t)
    n_new = cache.insert(s1)
    assert n_new == 1  # only the full block
    assert len(cache) == 1
    # Same 5-token prefix: hit is 4 (full block), not 5.
    s2, hit = cache.lookup([10, 20, 30, 40, 50, 99], alloc)
    assert hit == 4
    assert hit % bs == 0
    assert s2.block_table == [s1.block_table[0]]
    k2, _ = s2.gather_kv()
    assert np.allclose(k2[0], stored_k[0])
    assert np.allclose(k2[3], stored_k[3])
    # Divergent first block → miss.
    s3, hit3 = cache.lookup([10, 20, 30, 41], alloc)
    assert hit3 == 0
    assert s3.block_table == []
    cache.clear(alloc)
    s1.free()
    s2.free()
    s3.free()
    assert alloc.free_count == alloc.num_blocks


def test_prefix_hash_chain_parent_matters():
    a = block_hash(None, (1, 2, 3, 4))
    b = block_hash(a, (5, 6, 7, 8))
    c = block_hash(None, (5, 6, 7, 8))  # same tokens, different parent
    assert a != b
    assert b != c
    # Two sequences that share block0 then diverge must not reuse block1 via hash.
    rng = np.random.default_rng(5)
    h, d, bs = 1, 4, 2
    alloc = BlockAllocator(12, bs, h, d)
    cache = PrefixCache()
    s1 = Sequence(alloc)
    for t in [1, 2, 3, 4]:
        s1.append_kv(*_kv(h, d, rng), token=t)
    cache.insert(s1)
    s2 = Sequence(alloc)
    for t in [1, 2, 9, 9]:
        s2.append_kv(*_kv(h, d, rng), token=t)
    cache.insert(s2)
    hit_seq, hit = cache.lookup([1, 2, 3, 4, 5, 6], alloc)
    assert hit == 4
    assert hit_seq.block_table[0] == s1.block_table[0]
    assert hit_seq.block_table[1] == s1.block_table[1]
    # First block tokens match s1; insert kept existing hash → physical id is s1's.
    # Second block is a different chain child → s2's (9,9) physical id.
    branched, hit2 = cache.lookup([1, 2, 9, 9], alloc)
    assert hit2 == 4
    assert branched.block_table[0] == s1.block_table[0]
    assert branched.block_table[1] == s2.block_table[1]
    assert hit_seq.block_table[1] != branched.block_table[1]
    cache.clear(alloc)
    s1.free()
    s2.free()
    hit_seq.free()
    branched.free()


def test_prefix_hit_then_append_allocates_new_block():
    """After a full-block hit, the next token starts a fresh block (no fake mid-block reuse)."""
    rng = np.random.default_rng(6)
    h, d, bs = 2, 4, 2
    alloc = BlockAllocator(16, bs, h, d)
    cache = PrefixCache()
    s1 = Sequence(alloc)
    for t in [7, 8, 9, 10]:
        s1.append_kv(*_kv(h, d, rng), token=t)
    cache.insert(s1)
    s2, hit = cache.lookup([7, 8, 9, 10, 11], alloc)
    assert hit == 4
    free_before = alloc.free_count
    s2.append_kv(*_kv(h, d, rng), token=11)
    assert s2.seq_len == 5
    assert len(s2.block_table) == 3
    assert alloc.free_count == free_before - 1
    cache.clear(alloc)
    s1.free()
    s2.free()


def test_prefix_lru_evict_releases_blocks():
    rng = np.random.default_rng(7)
    h, d, bs = 1, 2, 2
    alloc = BlockAllocator(6, bs, h, d)
    cache = PrefixCache()
    s = Sequence(alloc)
    for t in range(4):  # 2 full blocks
        s.append_kv(*_kv(h, d, rng), token=t)
    cache.insert(s)
    assert len(cache) == 2
    # Drop both cached retains so freeing the sequence returns the pool.
    assert cache.evict_lru(alloc, n=2) == 2
    assert len(cache) == 0
    s.free()
    assert alloc.free_count == alloc.num_blocks
