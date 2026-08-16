"""CPU demo: allocate, fork/COW, paged attention."""

from __future__ import annotations

import numpy as np

from paged_kv import BlockAllocator, PrefixCache, Sequence, contiguous_attention, paged_attention

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    alloc = BlockAllocator(8, 4, 2, 8)
    s = Sequence(alloc)
    ks, vs = [], []
    for t in range(6):
        k = rng.standard_normal((2, 8)).astype(np.float32)
        v = rng.standard_normal((2, 8)).astype(np.float32)
        ks.append(k)
        vs.append(v)
        s.append_kv(k, v, token=t)
    child = s.fork()
    print("shared blocks", s.block_table, "refs", list(alloc.refcount[s.block_table]))
    child.append_kv(rng.standard_normal((2, 8)).astype(np.float32), rng.standard_normal((2, 8)).astype(np.float32))
    print("after COW    ", s.block_table, child.block_table)
    q = rng.standard_normal((2, 8)).astype(np.float32)
    print("attn match", np.allclose(paged_attention(q, s), contiguous_attention(q, np.stack(ks), np.stack(vs))))
    cache = PrefixCache()
    cache.insert(s)
    hit_seq, n = cache.lookup([0, 1, 1], alloc)
    print("prefix hit", n, "blocks", hit_seq.block_table)
