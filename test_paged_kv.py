"""Paged KV tests — fail if the memory model or attention is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from paged_kv import (
    BlockAllocator,
    KVCacheOOM,
    PrefixCache,
    Sequence,
    contiguous_attention,
    paged_attention,
)


def _kv(h, d, rng):
    return rng.standard_normal((h, d)).astype(np.float32), rng.standard_normal((h, d)).astype(np.float32)


def test_no_external_fragmentation():
    rng = np.random.default_rng(0)
    h, d, bs, nb = 2, 8, 4, 40
    alloc = BlockAllocator(nb, bs, h, d)
    live: list[Sequence] = []
    for _ in range(100):
        length = int(rng.integers(1, 18))
        s = Sequence(alloc)
        try:
            for _t in range(length):
                k, v = _kv(h, d, rng)
                s.append_kv(k, v)
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
    # Fixed-size blocks: remaining capacity is fully usable (no external holes).
    filler = Sequence(alloc)
    for _ in range(alloc.capacity_tokens):
        k, v = _kv(h, d, rng)
        filler.append_kv(k, v)
    assert alloc.free_count == 0
    filler.free()
    assert alloc.free_count == nb


def test_cow_share_then_diverge_refcount():
    rng = np.random.default_rng(1)
    h, d, bs = 2, 4, 4
    alloc = BlockAllocator(16, bs, h, d)
    s1 = Sequence(alloc)
    for _ in range(5):
        k, v = _kv(h, d, rng)
        s1.append_kv(k, v)
    # 5 tokens → blocks [full, partial]; both shared after fork.
    s2 = s1.fork()
    assert s1.block_table == s2.block_table
    last = s1.block_table[-1]
    first = s1.block_table[0]
    assert alloc.refcount[last] == 2
    assert alloc.refcount[first] == 2
    k, v = _kv(h, d, rng)
    s2.append_kv(k, v)  # writes the shared partial block → COW
    assert s1.block_table[0] == s2.block_table[0]
    assert alloc.refcount[first] == 2
    assert s1.block_table[-1] != s2.block_table[-1]
    assert alloc.refcount[s1.block_table[-1]] == 1
    assert alloc.refcount[s2.block_table[-1]] == 1


def test_paged_attention_matches_contiguous():
    rng = np.random.default_rng(2)
    h, d, bs = 3, 8, 3
    alloc = BlockAllocator(20, bs, h, d)
    # Interleave a second sequence so s's physical block IDs are not a contiguous run.
    s = Sequence(alloc)
    other = Sequence(alloc)
    ks, vs = [], []
    for i in range(bs * 3):
        k, v = _kv(h, d, rng)
        ks.append(k)
        vs.append(v)
        s.append_kv(k, v)
        if (i + 1) % bs == 0:
            other.append_kv(*_kv(h, d, rng))
    assert any(abs(s.block_table[i] - s.block_table[i + 1]) != 1 for i in range(len(s.block_table) - 1))
    q = rng.standard_normal((h, d)).astype(np.float32)
    k_c = np.stack(ks, axis=0)
    v_c = np.stack(vs, axis=0)
    out_p = paged_attention(q, s)
    out_c = contiguous_attention(q, k_c, v_c)
    assert np.allclose(out_p, out_c, atol=1e-5)
    assert s.block_table != sorted(s.block_table) or len(s.block_table) >= 1


def test_oom_when_blocks_exhausted():
    rng = np.random.default_rng(3)
    alloc = BlockAllocator(2, 2, 1, 4)
    s = Sequence(alloc)
    for _ in range(4):
        s.append_kv(*_kv(1, 4, rng))
    with pytest.raises(KVCacheOOM):
        s.append_kv(*_kv(1, 4, rng))


def test_prefix_cache_hit_reuses_blocks():
    rng = np.random.default_rng(4)
    h, d, bs = 2, 4, 2
    alloc = BlockAllocator(16, bs, h, d)
    cache = PrefixCache()
    s1 = Sequence(alloc)
    toks = [10, 20, 30, 40]
    stored_k = []
    for t in toks:
        k, v = _kv(h, d, rng)
        stored_k.append(k)
        s1.append_kv(k, v, token=t)
    cache.insert(s1)
    s2, hit = cache.lookup([10, 20, 30, 99], alloc)
    assert hit == 3
    assert s2.block_table[0] == s1.block_table[0]
    # Shared prefix physical blocks are the same objects.
    n_share = (hit + bs - 1) // bs
    assert s2.block_table[:n_share] == s1.block_table[:n_share]
    # Gathered K for the hit prefix matches the original.
    k2, _ = s2.gather_kv()
    assert np.allclose(k2[0], stored_k[0])
    cache.clear(alloc)
    s1.free()
    s2.free()
