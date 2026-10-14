# Baseline versus ReplaySSM — 2026-10-13

With matching runtime settings on the two DGX Sparks, the current ReplaySSM implementation was about 5% slower
on the measured decode workloads. Prefill was 1.1–1.5% slower, a small difference that needs more runs to establish.
All output token hashes matched. The current checkout was restored and left serving with KDA replay enabled.

## Versions and settings

| | Baseline | Current ReplaySSM |
|---|---|---|
| Checkout | `/home/dongsheng/retry/miaai/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` | `/home/dongsheng/retry/dgxRL/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` |
| Git revision | `3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab` | `b9bec69aae616affd26f99ae98c7edd155c81b51` plus local patch corrections |
| Docker image ID prefix | `9b00719301e8` | `4b3491f8a6e4` |
| Patch set hash | `e66c657a433d` | `f3fe6466ad02` |
| KDA replay | Absent | Enabled, window 16 |

These are distinct images, already available on both Sparks. The baseline image matches its checkout's pinned
published digest. Every baseline inference patch is byte-identical in the current checkout; the current image adds
the corrected `0109-glm-kda-replayssm.patch`. Saved metadata contains the full image IDs and launch settings.

Both runs used GLM-5.3-Flash EXL3 4 bpw, DFlash2 with policy `fnc7:0.3`, TP=2, PARALLEL=1, FP8 KV, q4 dense
weights, 1,048,576-token context, vision enabled, RoCE, split prefill, chunked KDA, 4,096-row prefill chunks, and
expert decode order 1. The relevant environment variables differed only in `TF_GLM_KDA_REPLAY=1` and
`TF_GLM_KDA_REPLAY_LEN=16` being present for the current image. Command-line arguments were identical.

The baseline's default parallelism and prefill tuning were overridden for this comparison. Its local configuration
file was not edited. The baseline was started from its own checkout with:

```bash
IMAGE=tensorfold-glm53:baseline-e66c657a433d PARALLEL=1 \
  TF_GLM_PREFILL_ROWS=4096 TF_GLM_EXL3_DEC_ORDER=1 ./start.sh
```

The current checkout was then restored using `./start.sh restart`, which reads its existing `scripts/local.sh`.
Both launchers passed the `OK` completion smoke test. Separate image tags preserved the current default image.

## Results

| Workload | Baseline tok/s | Replay tok/s | Replay change |
|---|---:|---:|---:|
| Code decode, 512 output tokens | 55.55 | 52.79 | −5.0% |
| Prose decode, 512 output tokens | 53.36 | 50.70 | −5.0% |
| Uncached prefill, 7,922 tokens | 2,052.64 | 2,029.09 | −1.1% |
| Uncached prefill, 30,344 tokens | 2,063.95 | 2,033.68 | −1.5% |

Each workload ran three times per image. The first repetition was a warm-up; the table divides token count by
the mean server-reported time of the remaining two. Decode uses `(completion_tokens - 1) / decode_s`; prefill
uses `prompt_tokens / prefill_s`. Decode used DFlash2; prefill used `draft=false` and 16 output tokens. All requests
used temperature 0 and thinking disabled. All 24 requests reported zero cached prompt tokens. Health counters
were checked around every request to detect concurrent completions; all checks passed.

Code decode ranges for the two measured repetitions were 55.53–55.57 tok/s baseline and 51.82–53.80 replay;
prose ranges were 53.14–53.59 and 49.63–51.83. These are short synthetic workloads measured baseline first,
then replay, without randomized repeated server restarts or controlled GPU clocks. They show no speed benefit
from replay in this test; they do not establish results for long-context decode or concurrent traffic.

## Correctness and memory

All repetitions and both images produced the same token hashes:

| Workload | Token hash |
|---|---|
| Code | `cdac1673a5e5` |
| Prose | `28196fee8d69` |
| 7,922-token prefill | `a5bb039a017e` |
| 30,344-token prefill | `9733f2f8efb8` |

Draft rounds and acceptance counts also matched between images for each decode workload: code 197 rounds,
315 accepted / 478 drafted; prose 220 rounds, 292 accepted / 465 drafted.

Both runs requested a 12.5 GiB retained-prompt cache and 14.5 GiB memory reserve. Startup admitted about
11.9 GiB of retained-prompt cache in both runs. Minimum sampled host `MemAvailable` was 18.75 GiB during
baseline measurements versus 14.78 GiB with replay, about 4 GiB less headroom. This is host-wide available
memory, not a direct allocation measurement of replay buffers. The test did not fill the prompt cache or exercise
the full 1M context window.

The earlier [performance tuning report](PERFORMANCE_TUNING_20261013.md) compared two tuning configurations
with replay enabled in both; its “baseline” is different from the original miaai checkout tested here.

## Artifacts

[Benchmark driver, raw JSONL measurements, image metadata, and startup logs](benchmarks/20261013-replayssm/)
are retained in the repository. The driver records each request immediately under `/tmp/glm-perf-LABEL-PHASE.jsonl`;
use a new label on reruns because it appends. Run each phase after the launcher has finished its smoke test:

```bash
python3 docs/benchmarks/20261013-replayssm/benchmark.py draft --label NEW_LABEL --policies auto
python3 docs/benchmarks/20261013-replayssm/benchmark.py prefill --label NEW_LABEL
```
