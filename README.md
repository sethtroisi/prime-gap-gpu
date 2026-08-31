# prime-gap-gpu - a new GPU program to find prime gaps.

A fast prime gap searching tool.

# Table of Contents

- [Overview](#overview)
- [Setup](#setup)

## Overview

TBD

## Setup

In general this is going to be easy under Ubuntu 24.04 or later

```bash
$ sudo apt install libgmp10 libgmp-dev
$ sudo apt install build-essential automake autoconf make
# CUDA is required but I'm not sure what apt install that is
```

```
$ sudo apt install libmpfr-dev libmpc-dev
$ python -m pip install --user gmpy2
```

```
$ git clone https://github.com/sethtroisi/prime-gap-gpu.git
$ cd prime-gap-gpu
```

```
make

or

(Not reccomeneded but here for clang-tidy maybe)
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBITS=512
cd build
make
```


## Misc

```
valgrind --suppressions=cuda.supp --leak-check=full ./gap_search_gpu -p 337 -d 2310 --mstart 10000000 --minc 200000 --max-prime 1 --min-merit 25 -v -v -v
```

## TUNING

  * `-p` prime (AKA `log(K)`)
    * Decreasing means faster PRP and fewer PRP/m so `O(n^2)` if move below a threshold of 512, 768, 1024 bits.
    * First order effect of decreasing is CPU overflow may bog down more.
  * `--cpu-fraction`
    * Increasing leads to less sparse sieves across X, trades off for more overflow work
    * Should lower (more sieving) till `Waiting 4 sieves` becomes 5-10%.
    * `--cpu-fraction` **doesn't (significantly) change the total number of PRP tests**.
      It moves PRP tests from main testing thread to overflow thread.
      This has some change in sieve level (bad) and some decrease in `waiting 4 sieves` (good).
  * `OVERFLOW_SIEVE_LIMIT`: TODO
    * Trades CPU sieving for GPU time, look at `total time     : sieve` from `CPU OVERFLOW Timing`
  * overflow.cpp: `stop_x`
    * Increasing leads to less numbers running out of the sieved range (and overflowing to CPU)
    * Decreasing leads to faster sieving.
  * `max-prime` better to increase at some point top primes never run
  * `minc`: TODO add some metric to tune on.
  * `min-merit` math | tuned ~5% past optimal. Reduces overflow by 3x at cost of 4-8% overall efficency.

These are likely set to good values

   * `OPEN_SIEVES` 3-6 is probably great balance of enough unknown count smoothing while minimizing memory usage.
   * `GPU_BATCHES` 2-3, 3 is probably better.
   * `WINDOW_BITS` 4-5, 4 for 256, 5 for 512

## Upgrades

  * Have `sieve_interval_cpu` do both directions, and do GPU offloading of `prev_prime`.
    * Pros:
      * Would free up 5+ CPU cores
      * Allows for setting lower `min_gap_to_continue` which is 5% more optimal.
    * Cons:
      * This is a fixed amount of work (doesn't change with `--cpu-fraction`)
  * Consider choosing a consistent X to overflow at.
    * Pros:
      * If known before hand might simplify some of the CPU overflow sieve math & tracking
      * Can start sieves for next range ahead of time.
      * Don't have to track `sieve_start` per overflow
    * Cons:
      * Less dynamic flexibility
  * On 2026/08/31 most time was spent in these places:
    * GPU Timing: 260% running, very low waiting 4 sieve, 30% wait done.
      * Wait done is `run_testing_thread`, `increment_X` & `push_to_overflow`.
        * **Measured as 5s/ reset which seems like it could be optimized slightly**
    * GPUSieve
      * 50/50 in small and large kernels. Seems great!
      * **There are known large kernel optimizations** that could be tried
    * CPUSieve
      * 30% spent in finalize, **would be nice to reduce**, not sure how.
      * This is mostly CPU time but could help reduce "wait 4 sieve" and possibly "wait done"
    * Overflow:
       * 99% on GPU. 9K in sieve, 45K in `prev_prime`.
       * Requires 6 CPU workers, could go to 3 **if `prev_prime` was handled on GPU.**
       * <1% of total prime test.

## TODO

  * [ ] Faster GPU sieving
  * [ ] Is there a way to tricker overflow GPU testing only when sieving is "small"
    * Wait till `max_p_i` is reduced, fire a signal, overflow runs till empty.
    * This shifts work so that GPU can be more full when main testing is sparser

## TODONE

  * [x] Wheel for divisors of d.
  * [x] Check a small percent of next_primes are actually prime
  * [x] Understand what sieve limit gmp is using for overflow
    * GMP is sieving to ~4M -> 10% less efficient
  * [x] Try changing vector<uint8_t> to vector<uint32_t>
    * This didn't seem to have any impact on speed, but I was told it might help in
      reduce mixed cache line access (8 vs 32 vs 64) control so I'll keep it.
  * [X] Offload overflow `probab_prime` back to the GPU
    * Sieve each m range, keep batch of these sieves
    * Make a `GPUBatch` of sieves and current index into `coprime_X`
  * [ ] Understand why some X=12 have twice as many unknowns?
    * `m * K % 3` is 1 or 2; and doesn't remove any factors from `X=12`
    * When `X % 3 == 0` you end up with twice as many factors.
    * With `X % 3 == {1, 2}` half of factors get removed by 3, (1/4 with 5, 1/6 with 7)
