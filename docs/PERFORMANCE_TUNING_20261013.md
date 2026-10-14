# Two-Spark performance tuning — 2026-10-13

Local measurement on `mydgx0` and `dongsheng@10.100.136.2`, NVIDIA GB10, patched TensorFold image `f3fe6466ad02`.
The model is GLM-5.3-Flash EXL3 4 bpw, q4 dense weights, FP8 KV, DFlash2, RoCE, split prefill, vision enabled,
one request at a time, 1,048,576-token context, KDA chunking and replay enabled with replay window 16.

## Settings retained

```bash
KDA_REPLAY=1
PARALLEL=1
export TF_GLM_PREFILL_ROWS=4096
export TF_GLM_EXL3_DEC_ORDER=1
```

These installation-specific settings are in `scripts/local.sh`. The draft policy remains `fnc7:0.3`,
`MEMORY_RESERVE_GIB` remains 14.5, and `KV_POOL_GIB` remains 12.5. No model quantization change.

## Prefill

Three sequential requests per prompt with temperature 0, thinking disabled, `draft=false`, and 16 output tokens.
Every request reported zero cached prompt tokens. The first request warms kernels; the table uses the mean
prefill time of the following two. Throughput is prompt tokens divided by server-reported prefill seconds.

| Prompt tokens | 2,048-row chunks | 4,096-row chunks | Throughput before → after | Change |
|---:|---:|---:|---:|---:|
| 7,922 | 4.115 s | 3.888 s | 1,925 → 2,037 tok/s | +5.8% |
| 30,344 | 15.958 s | 14.920 s | 1,902 → 2,034 tok/s | +7.0% |

Inputs contain 480 or 1,800 records generated with:

```python
prompt = '\n'.join(f'Record {i}: warehouse has {i%97} blue bolts and {i%83} green nuts.' for i in range(lines))
prompt += '\nSummarize the record format in one sentence.'
```

## Decode

Two 512-token greedy completions per workload and policy. Decode throughput is `(completion_tokens - 1) / decode_s`.
Baseline policies were tested in forward and reverse order. The API `tf_policy` field selects the policy per request.

| Policy | Code tok/s | Prose tok/s |
|---|---:|---:|
| auto | 51.37 | 49.51 |
| fcost7:noisy | 50.96 | 48.37 |
| f7 | 38.78 | 35.40 |
| fnc7:0.15 | 50.50 | 47.93 |
| fnc7:0.5 | 50.55 | 48.81 |

`auto` resolves to the configured `fnc7:0.3` DFlash2 policy for this server. It was the best policy tested.

| Current policy | Before | After 4,096-row chunks + expert order 1 |
|---|---:|---:|
| code | 51.37 tok/s | 51.74 tok/s |
| prose | 49.51 tok/s | 49.90 tok/s |

The decode difference is less than 1%; it should be treated as within measurement noise, not a demonstrated speedup.

Code prompt:

> Write a complete Python standard-library implementation of an LRU cache with TTL, thread safety, a fake clock, and unittest tests. Explain the invariants in comments.

Prose prompt:

> Write a detailed practical guide to maintaining a bicycle, including cleaning, inspecting brakes, adjusting gears, and fixing a flat. Use complete explanatory paragraphs.

## Correctness and memory

Output token hashes matched across all policies and both configurations for each workload.
Baseline hashes: code `cdac1673a5e5`, prose `28196fee8d69`, 7,922-token prefill `a5bb039a017e`,
30,344-token prefill `9733f2f8efb8`. Both restarts passed the launcher smoke test.

Geometry estimates 1.724 GiB of additional working memory per rank for 4,096-row prompt buffers.
The tuned run had at least 15.53 GiB MemAvailable on the head during these tests. The worker had about 16 GiB
available immediately after startup. Swap usage did not increase materially.

The second startup admitted the full 12.5 GiB prompt-reuse cache: rank 0 estimated 89.14 GiB against a 101.72 GiB
budget; rank 1 estimated 87.34 GiB against 101.85 GiB. The first run admitted 9.6 GiB because less memory was
available at that startup. This cache increase reflects startup headroom, not a benefit of larger prompt chunks.

The cache allowance is a budget for retained conversations, not a measured filled cache. Full 1M-context and
full prompt-cache pressure were not tested. These results cover two synthetic decode workloads and prompts up
to 30,344 tokens; they do not establish a general speedup for all workloads.

Raw measurements and the benchmark driver from this session are under `/tmp/glm-perf-*`.
