"""CPU demo: block table, fork/COW, hash prefix cache, paged attention."""

from __future__ import annotations

import numpy as np

from paged_kv import (
    BlockAllocator,
    PrefixCache,
    Sequence,
    contiguous_attention,
    paged_attention,
    paged_attention_blockwise,
)

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
    child.append_kv(
        rng.standard_normal((2, 8)).astype(np.float32),
        rng.standard_normal((2, 8)).astype(np.float32),
        token=6,
    )
    print("after COW    ", s.block_table, child.block_table)
    q = rng.standard_normal((2, 8)).astype(np.float32)
    out_g = paged_attention(q, s)
    out_b = paged_attention_blockwise(q, s)
    out_c = contiguous_attention(q, np.stack(ks), np.stack(vs))
    print("attn match gather/blockwise", np.allclose(out_g, out_c), np.allclose(out_b, out_c))
    cache = PrefixCache()
    n_new = cache.insert(s)
    hit_seq, n = cache.lookup([0, 1, 2, 3, 1], alloc)
    print("prefix full-block hit", n, "new_hashes", n_new, "blocks", hit_seq.block_table)
    hit_seq.free()
    cache.clear(alloc)
    child.free()
    s.free()
    print("pool restored", alloc.free_count == alloc.num_blocks)
