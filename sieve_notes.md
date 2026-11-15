# GPU sieve optimization notes

Goal: speed up `GPUSieve::run()` by 50-100% for

```
-p 353 -d 2310 --minc 50'000'000 --max-prime 1000      (--prime-limit is an alias of --max-prime)
```

All numbers are from a Colab **NVIDIA L4** (sm_89, 58 SMs) unless noted. No local GPU was available,
everything was run through `colab_benchmark.sh`.

## Summary

| (L4, target config) | ms / sieve | speedup |
|---|---|---|
| baseline (6fb5ac2) | 7.23 | |
| v2: segmented sieve in shared memory + bit packed composite | 2.51 | 2.9x |
| v3: + batch of 4 X per pass over the large primes (default on L4) | 1.32 | 5.5x |

T4: 40.0 -> 7.0 ms (batching is off by default there). Details, caveats and what didn't work below.

Round 2 (max-prime 2000, GPU finalize), L4, sieve + finalize with 25% of coprime m active:

| | max-prime 1000 | max-prime 2000 |
|---|---|---|
| baseline sieve only (no finalize) | 7.23 | 9.1 |
| v3 sieve + CPU finalize (what gap_search_primorial did) | 6.59 | 6.96 |
| sieve + GPU finalize (`run_unknowns`), batch 4 | 1.65 | 2.10 |
| sieve + GPU finalize, batch 8 | | 1.81 |

T4 at 2000: 15.6 -> 11.4 ms.

Reproduce: `./colab_benchmark.sh --tag <label>` (defaults to the target config on an L4, add `--stop` to release the session).

## Tooling

* `gpu_sieve_benchmark.cu`: runs `GPUSieve::run()` for `--sieves` X values over `--ranges` m ranges.
  * verifies `--verify` sieves per range bit-for-bit against a simple (independent) CPU sieve.
  * prints a checksum over every sieve result so builds can be compared.
  * `--mstart` defaults to 80e9.
* `run_sieve_benchmark.sh`: build (auto detects `ARCH` and `MP`) + run + append CSV row.
* `colab_benchmark.sh`: tar repo (tracked + untracked non-ignored), upload to Colab, run the above, download CSV row
  into `sieve_benchmark_results.csv`.
  (`git_hash` in the existing rows refers to local commits that were squashed before pushing.)
* `misc/gpu_sieve_experiments.cu` (`make gpu_sieve_experiments`): micro benchmarks of alternative kernels,
  each verified against a CPU sieve of the same primes.

## Workload shape (target config)

* `BITS = m_inc / 2 = 25M` odd m per sieve, 88.1% end up composite.
* 50,847,463 primes: 4 d-primes (wheel) | "small" 2 * GRID = 464 primes (359 .. ~3.7K)
  | "medium" 1.56M primes < 25M | "large" 49.3M primes in [25M, 1e9]
* writes per sieve ~ `BITS * (lnln(hi) - lnln(lo))`: small ~8.6M, medium ~18M, large ~4.9M.

## Baseline (6fb5ac2)

7.2 ms / sieve (138 sieves/s)

| phase  | ms    | what |
|--------|-------|------|
| wheel1 | 0.08  | host builds wheel, H2D copy |
| wheel2 | 0.23  | wheel tiled into byte array |
| small  | 0.58  | 464 primes, 1 block per prime |
| medium | 3.43  | 1.56M primes, strided byte writes into 25MB |
| large  | 2.35  | 49.3M primes, int64 `%` each, ~4.9M writes |
| copy   | 0.58  | bytes -> bits kernel + 3MB D2H |

* large scales linearly with prime count (max-prime 100 -> 0.27 ms): 48 ps / prime = 168 GB/s of (prime, inv) reads.
  Suspect memory bandwidth bound more than compute bound.

## Experiments (`misc/gpu_sieve_experiments.cu`)

L4 facts: 48MB L2 cache, 48KB shared memory per block (99KB opt-in).

### Small / medium primes (writes dominate)

| experiment | ms |
|---|---|
| baseline small (464 primes, bytes) | 0.49 |
| baseline medium (bytes, 12.9M marks) | 3.36 |
| baseline compress bytes -> bits | 0.30 |
| medium, `atomicOr` into bit packed array | 2.01 |
| small+medium `atomicOr` in one strided kernel | 6.56 (bad: tiny primes hammer the same words) |
| **segmented, bytes in shared mem, S=64K, 1024 threads, p < 64K** | **0.08** |
| **segmented, bytes in shared mem, S=64K, 1024 threads, p < 256K** | **0.17** |
| segmented, `atomicOr` bits in shared mem, S=256K, p < 256K | 0.37 |
| medium `atomicOr` bits, only p >= 64K / 256K / 1M | 0.49 / 0.32 / 0.21 |
| medium bytes, only p >= 64K / 256K / 1M | 0.68 / 0.42 / 0.26 |

* Random byte writes into the 25MB array are slow, the same marks into a 3MB bit array with `atomicOr` are ~35% faster.
* The big win: each block owns 64K consecutive flags as **bytes in shared memory** (no atomics), sieves all
  primes < 4 * 64K, then packs bytes to bits (multiply trick), ORs in the wheel and writes whole words.
  This replaces wheel + small + most of medium + compress.
* Shared memory atomics are ~2-4x slower than plain byte stores to shared memory, 256 threads / block is slower
  than 1024 for the 64K segment.

### Large primes (memory bandwidth dominates)

| experiment | ms |
|---|---|
| baseline (int64 %, byte write) | 2.28 |
| read only, primes + inverses (8B / prime) | 1.58 |
| read only, primes only (4B / prime) | 0.82 |
| read only, packed uint64 (prime, inv) | 1.52 |
| int64 % + `atomicOr` bits | 1.64 |
| float reciprocal + `atomicOr` bits | 1.61 |
| float, primes as uint8 half gaps (5B / prime), chunks of 32 | 1.72 (worse!) |
| launch shape (blocks x threads) | no effect, 1.52 - 1.56 |
| batch of 2 / 4 / 8 X per pass (float math) | 1.61 / 2.81 / 5.62 total => 0.81 / 0.70 / 0.70 per sieve |
| batch of 2 / 4 / 8 X per pass (int64 %) | 1.82 / 4.08 / 8.11 total |

* Reading (prime, inverse) is ~400MB per sieve => ~260GB/s, at the L4's memory bandwidth. The modulo is hidden behind it.
* So: sieve several X per pass over the primes and make the per X math cheap.
* D2H copy of the 3MB bit array: 0.25 ms from pinned memory (0.82 pageable).

## v2: segmented sieve + bit packed composite

* `segment_kernel`: wheel + primes < 256K, 382 blocks x 1024 threads, 64KB shared memory (opt-in, falls back to 32K on
  GPUs with less).
* `medium_primes_kernal` / `large_primes_kernal`: unchanged math, `atomicOr` into bits.
* wheel is uploaded as `wheel_size` (= product of odd primes of d = 1155) 32 bit windows.
* no compress kernel, per kernel syncs only with `set_profile(true)` (verbose >= 2 or `--profile`).

**2.48 ms / sieve (2.9x)**, checksum identical to baseline.

| phase (with --profile, 2.7 ms total) | ms |
|---|---|
| wheel   | 0.05 |
| segment | 0.27 |
| medium  | 0.39 |
| large   | 1.71 |
| copy    | 0.32 |

## v3: batch several X per pass over the large primes

* `large_primes_batch_kernal`: reads (prime, inverse) once and sieves up to `MAX_SIEVE_BATCH = 8` X values into
  separate composite arrays.
  * `-M_start % p` once per prime (uint64 %), then per X only 32 bit math:
    `q = (int)(float(X) * (float(inv) / float(p)) - 0.05)`, `r = X * inv - q * p` (mod 2^32, in `[0, 2p)`), two `min(r, r - p)`.
    Needs `X < 2^17` (float error < 0.04), otherwise falls back to the exact per X `large_primes_kernal`.
* `GPUSieve::run(m_start, m_inc, X, max_p_i)` API is unchanged. On a cache miss it sieves X and the next
  `batch_size - 1` coprime X (from `get_coprime_X`), later calls with the same `m_start`, a cached X and
  `max_p_i <=` what was used just copy the result back (0.3 ms).
* `set_batch_size(n)`, 1 disables batching. Default is `clamp(L2 cache / (2 * composite bytes), 1, 4)`
  (4 on L4, 1 on T4, see T4 section).

Behaviour changes to be aware of:

* speculative: up to `batch - 1` sieves are wasted when the caller moves to the next m range.
* latency: first call of a batch costs the whole batch (~4.4 ms for 4, ~8.1 ms for 8), the rest ~0.3 ms.
* a cached sieve may have used *more* primes than the `max_p_i` of the later call (never fewer).
* GPU memory: 8 composite arrays (24MB instead of 25MB+3MB before).

L4, target config, average over all 80 sieves (2 ranges x 40 X, after an untimed warm-up):

| batch | ms / sieve | vs baseline 7.23 |
|---|---|---|
| 1 | 2.51 | 2.9x |
| 2 | 1.69 | 4.3x |
| **4 (default)** | **1.32** | **5.5x** |
| 8 | 1.30 | 5.6x |

Breakdown at batch 4 with `--profile` (1.48 ms total, the extra syncs cost ~0.15 ms):
wheel 0.02 | segment 0.27 | medium 0.40 | large 0.53 | copy 0.33

* batch 8 barely beats 4: per X math + `atomicOr` (~0.4 ms / X) now dominates the shared 1.5 ms read.

## Correctness

* Checksum over all 80 benchmark sieves is identical (`efd68c8b242a2a19`) for baseline, v2 and v3 at every batch size.
* Benchmark verifies sieves bit-for-bit against an independent CPU sieve (first + last X of each range).
* 13 other configs verified (`--sieves 12 --ranges 2 --verify 4`): p=151/337/353/503/1009, d=2/210/2310/30030/510510,
  minc 100K .. 120M (including `m_inc/2` odd), mstart 2 .. 9.99e12, max-prime 1 .. 2000, `--max-pi` 1000 / 50000 / 3M,
  batch 3 / 5 / 8.
* Full program: `gap_search_primorial -p 337 -d 2310 --mstart 81580000000 --minc 40000000 --max-prime 100 --min-merit 14`
  for 100 s on L4 (BITS=512), baseline (6fb5ac2) vs v3: identical `records_337.txt` (6884 gaps, same md5 after sort),
  identical total PRP tests (412,704,309).

## Effect on the full search (L4, config above)

| | baseline | v3 |
|---|---|---|
| GPU sieve / sieve (sharing the GPU with PRP tests) | 6.6 ms | 2.24 ms |
| sieve thread total time | 15.0 s | 8.1 s |
| finalize (CPU) | 4.8 s | 4.7 s |
| m processed / s | 406,927 | 405,790 |

* **No end-to-end change for this config**: the sieve thread wasn't the bottleneck ("waiting 4 sieve" 0%).
* CPU `finalize` (3.1 ms / sieve) is now larger than the GPU sieve; it's the next thing to look at if sieving
  ever limits the search (deeper `--max-prime`, lower `--cpu-fraction`).
* Not measured: the full search at the target config (`-p 353 --max-prime 1000`).

## T4 (sm_75, 40 SMs, 6MB L2)

| | ms / sieve |
|---|---|
| baseline | 40.0 |
| v3 batch 1 | 6.9 (5.8x) |
| v3 batch 4 | 8.7 |
| v3 batch 8 | 9.2 |

* Baseline is much worse than on L4: the 25MB byte array doesn't fit the T4's L2 cache.
* Batching *hurts* on T4. With `--profile` at batch 4: segment 0.86 | medium 1.45 | large 6.65 | copy 0.34.
  Best guess: 4 x 3MB composites don't fit in the 6MB L2 so the random `atomicOr` miss (not confirmed with an experiment).
* => default batch size is picked from the L2 cache size. Only checked on these two GPUs, worth re-checking
  (`--batch 1/2/4/8`) on anything else.

## max-prime 2000

(see the next section, the 31 bit audit below missed that `2^31 <= prime` was broken)

98,222,216 primes (96.7M "large"), largest < 2^31 so `2 * prime < 2^32`.

* gpu_sieve.cu audit: primes are `uint32_t`, inverses `int32_t` (< prime < 2^31).
  `first_composite_index` uses int64. `large_primes_batch_kernal` needs `r < 2 * prime < 2^32`,
  `r + neg_M_start <= 2 * prime - 1`, `mi_0 + prime < 2 * prime`: all fine for prime < 2^31.
  Nothing needs `3 * prime`. `invert()` coefficients are bounded by prime.
* Benchmark at `--max-prime 2000` verifies bit-for-bit against the CPU sieve (which uses 64 bit math everywhere).
* gap_search_primorial.cpp had two limits:
  * rejected `--max-prime` > 1500. Now 2000 (same as `Args`).
  * `prime_count = max_prime / log(max_prime)` (93.4M at 2e9, 48.3M at 1e9) was passed as `max_p_i` so the
    largest ~5% of primes were never used. Now `gpu_sieve.get_num_primes()`.
* L4 sieve only (no finalize): baseline 9.1 ms -> 4.36 (batch 1) / 1.77 (batch 4, default) / 1.61 (batch 8).
  Profile at batch 4: segment 0.27 | medium 0.40 | large 1.00 | copy 0.33.

## Primes up to 2^32 (`--max-prime 4294`)

The first version broke above 2^31: `invert()` and the inverse arrays were `int32_t` and
`large_primes_batch_kernal` needed `2 * prime < 2^32`.

* `invert()` is int64, inverses are `uint32_t` on host and device.
* `large_primes_batch_kernal`: `r = X * neg_inv_K - q * prime` (in `[0, 2 * prime)`) is computed in uint64,
  the subtraction of `M_start mod p` and the "first odd mi" step are arranged to stay below `prime`.
* `Args` / gap_search_primorial accept `max_prime < 2^32`, the CPU sieve path (`GPU_VERIFY`) got the same int64 `_invert`
  and uint64 loop counters (only compile checked).
* misc/gpu_sieve_experiments.cu still needs `2 * prime < 2^32` (asserts max prime <= 2000).

Verified on L4 (bit-for-bit against the CPU sieve, which is 64 bit everywhere):

| config | result |
|---|---|
| target config, max-prime 1000 / 2000 | same checksums as before, 1.35 / 1.81 ms (was 1.31 / 1.77, the uint64 `r`) |
| max-prime 4000, `--finalize gpu` (189.9M primes) | PASS, 3.09 ms / sieve |
| max-prime 4294 (203.2M primes), batch 1 / 4 / 8, mstart 8e10 and 9.95e12 | PASS |
| `-p 151 --minc 2000000 --mstart 2 --max-prime 4294` | PASS |
| `-p 13109 --max-prime 2500 --first-x 131072` (X >= 2^17 so the exact `large_primes_kernal` runs) | PASS |
| gap_search_primorial `--max-prime 4000`, 140 s | same records as baseline for the 4 m ranges it finished |

GPUSieve setup is ~55 s and 1.5GB of GPU memory at 4e9 (single threaded inverses, 8 bytes / prime).

## GPU finalize (`GPUSieve::run_unknowns`)

The sieve thread used to copy the composite bits back and loop over `active & ~composite` on the CPU
while holding `sieve_mtx` (3.1 ms / sieve in the full program, ~5 ms in the benchmark with 336K unknowns).

* `unknown_count_kernel` / `unknown_scan_kernel` / `unknown_write_kernel`: popcount per 1024 bit chunk,
  prefix sum (one block), write sorted `mi = 2 * bit + 1`.
* Active bits are uploaded once per batch (`wants_active_bits()` / `set_active_bits()`, a 3MB memcpy under the lock).
  Results for the rest of the batch use slightly stale active bits, which is fine because
  `try_set_testing_data` already removes found primes (`remove_vector`) and `found_prime_m_i` is cumulative per m range.
* Unknowns for the whole batch are copied to pinned memory at once (one sync for the counts, one for the copies),
  so the other `run_unknowns` calls of the batch don't touch the GPU.
  * Tried copying a guessed count before the sync to save one sync per batch, no measurable gain, removed.
* More than `m_inc / 16` unknowns (capacity) falls back to copying composite + CPU loop.
* Checked with `--finalize gpu` on 13 configs (list must equal CPU decode of the full composite & active for every sieve),
  including ones that hit the capacity fallback (d=2 or `--max-pi 1000` with every coprime m active).
* gap_search_primorial.cpp: unknowns are staged outside the lock, under the lock it's just a vector swap.

L4, `-p 353 -d 2310 --minc 50M --max-prime 2000`, sieve + finalize per sieve:

| active fraction (unknowns / sieve) | run() + CPU finalize | run_unknowns() |
|---|---|---|
| 0.25 (336K) | 6.97 ms | 2.15 ms |
| 1.00 (1.34M) | 10.55 ms | 3.38 ms |

(the full program averaged 9% of m active in the test below, so 0.25 is on the pessimistic side)

Full program (L4, 100 s, `-p 337 -d 2310 --mstart 81580000000 --minc 40000000 --min-merit 14`):

| | baseline, max-prime 100 | new, 100 | new, 2000 |
|---|---|---|---|
| records_337.txt (6884 gaps) | | identical | identical |
| sieve thread total | 14.7 s | 5.3 s | 7.7 s |
| CPU finalize | 4.6 s | 0 | 0 |
| total PRP tests | 412.7M | 411.3M | 353.8M |
| m processed / s | 411,897 | 414,112 | 466,103 |

* Inside the full program the GPU side of a sieve is slower than in the benchmark (~3.2 ms / sieve at max-prime 100)
  because the GPU is busy with PRP tests. Fewer kernel launches per batch (one launch for all X of a batch) might help
  the latency, it wouldn't reduce GPU work. Not tried.

## What limits the large prime kernel now (L4, 2e9, batch of 8 = 7.1 ms)

| variant | ms |
|---|---|
| production (atomicOr) batch 1 / 4 / 8 | 3.37 / 4.30 / 7.09 |
| count hits instead of writing, batch 1 / 4 / 8 | 3.44 / 3.55 / 4.46 |
| racy load / or / store instead of atomicOr, batch 8 | 8.46 |
| only atomicOr if the bit isn't already set (86% set), batch 8 | 6.73 (vs 6.72) |
| only atomicOr if the bit is in a 3MB active mask, density 0.42 / 0.25 / 0.10 | 7.59 / 7.22 / 6.76 |
| load 2 / 4 / 8 primes per thread before any math | 7.2 / 7.8 / 7.5 |
| launch shape (2..16 x MP blocks, 128..1024 threads) | 6.8 - 8.5, 4 x MP x 256 is fine |
| transposed gaps (5B / prime, see below) batch 1 / 4 / 8 | 2.37 / 4.41 / 6.87 |

* Reading the primes is 3.4 ms, the math for 8 X ~1 ms, the 46M random bit sets ~2.6 ms.
* It's the random access that costs, not the atomic: a read of a same sized mask costs the same,
  so skipping inactive m or already composite bits doesn't help.
* Transposed gap layout: super blocks of `threads * 64` primes, thread g owns 64 consecutive primes (a column),
  stored row major so every load is coalesced. `first prime + uint8 half gaps + inverse` = 5.06 bytes / prime.
  Fixes what was wrong with the first gap experiment: 3.37 -> 2.37 ms at batch 1 (and 300MB less GPU memory at 2e9),
  but no gain at batch >= 4 where reads aren't the limit. Not put in gpu_sieve.cu, would be worth it for batch 1 / 2.
* `--batch 8` is ~9% faster than 4 at 2e9 (1.61 vs 1.77), no difference at 1e9. Default left at 4 (latency, staleness).

## Ideas not done

* ~0.2 ms / sieve is now a memcpy of the unknowns from pinned memory into the `next_sieves` vector.
  Avoiding it needs the vectors themselves to be pinned (allocator) so the D2H copy can land in them.
* One kernel launch per batch (instead of per X) for segment / medium / finalize, see GPU finalize section.
* Remaining L4 time at batch 4 is spread evenly: segment 0.27, medium 0.40, large 0.53, copy 0.33.
  * medium: raise `SEGMENT_PRIME_MULT` / tune the split between segment and `atomicOr` kernels (0.17 + 0.32 measured
    in isolation for a 256K split, 0.08 + 0.49 for 64K).
  * large: find out what the ~0.4 ms / X in the batch kernel is (atomics vs math), gap compressed primes were slower
    but the access pattern (per thread chunks of 32) was probably the problem not the idea.
  * copy: 3MB D2H per sieve, would shrink with the finalize idea above.
* `GPUSieve()` setup is 13.7 s single threaded for 50.8M primes (the experiments binary does the same work in 2 s with
  `primesieve::generate_primes` + OpenMP).
* Batch size heuristic is only validated on L4 and T4.
* `GPU_VERIFY` in gap_search_primorial.cpp can't be exact: CPU and GPU `max_p_i` index different prime lists
  (pre-existing TODO), and with batching a cached sieve may use more primes than requested.
