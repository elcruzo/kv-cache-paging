# Paged KV cache (vLLM block-table memory model)

Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023).

**2026 note:** vLLM deleted the named `PagedAttention` CUDA kernel (PR [#47361](https://github.com/vllm-project/vllm/pull/47361), July 2026) but **kept the block-table memory model**. This folder implements that model — fixed-size physical blocks, per-sequence block tables, copy-on-write sharing, and hash-chained automatic prefix caching — plus attention over non-contiguous blocks. It is not a CUDA kernel and not a vLLM/FAISS wrapper.

## Memory model

- `BlockAllocator`: free list of physical blocks, each holding `block_size` tokens of K and V. Fixed block size ⇒ **no external fragmentation**.
- `Sequence.block_table`: logical block index → physical block id. Token `t` lives at `(block_table[t // B], t % B)`.
- Allocate on demand; `free()` drops refcounts and returns empty blocks to the pool.
- **Copy-on-write:** `fork()` increments refcounts. A write into a shared block clones it (`cow`: `exclusive_write` vs `clone_shared`) so siblings stay intact.

## Attention

Two named paths (same math, no silent switch):

- `paged_attention` — gather K/V in logical order via the block table, then scaled-dot-product.
- `paged_attention_blockwise` — online softmax while walking physical blocks (no full gather).

## Prefix cache (APC)

`PrefixCache` indexes **full blocks only** by `sha256(parent_hash || block_tokens)`. Lookup walks the chain until the first miss; hit length is always a multiple of `block_size`. Partial trailing blocks are never reused. LRU eviction drops oldest hash entries and releases the cache retain.

## Papers on disk

- [`papers/kwon-vllm-pagedattention-2023.pdf`](papers/kwon-vllm-pagedattention-2023.pdf) — Kwon et al. Efficient Memory Management for LLM Serving (2023) ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180))

## Run

```bash
python demo.py
python -m pytest test_paged_kv.py -q
```
