# Paged KV cache (vLLM memory model)

Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023).

**2026 note:** vLLM deleted the named `PagedAttention` CUDA kernel (PR [#47361](https://github.com/vllm-project/vllm/pull/47361)) but **kept the block-table memory model**. This folder implements that model plus attention over non-contiguous blocks — not a CUDA kernel.

## Memory model

- `BlockAllocator`: free list of physical blocks, each holding `block_size` tokens of K and V. Fixed block size ⇒ **no external fragmentation** (any free block can satisfy any allocate).
- `Sequence.block_table`: logical block index → physical block id. K/V for token `t` live at `(block_table[t // B], t % B)`.
- Allocate on demand as the sequence grows; `free()` drops refcounts and returns empty blocks to the pool.
- **Copy-on-write:** `fork()` increments refcounts (beam / prefix share). A write into a shared block clones it (`cow`) so siblings stay intact.

## Attention

`paged_attention` gathers K,V **in logical order via the block table** (physical ids need not be contiguous) and computes standard scaled-dot-product attention against Q. Gather is the point — the math is ordinary attention.

## Prefix cache

`PrefixCache` maps an exact token-prefix to a retained block-table snapshot. A later sequence with the same prefix **reuses the same physical block IDs** until it writes (COW).

## Papers on disk

- [`papers/kwon-vllm-pagedattention-2023.pdf`](papers/kwon-vllm-pagedattention-2023.pdf) — Kwon et al. Efficient Memory Management for LLM Serving (2023) ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180))

## Run

```bash
python demo.py
python -m pytest test_paged_kv.py -q
```
