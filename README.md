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

These are likely set to good values

   * `OPEN_SIEVES` 3-6 is probably great balance of enough unknown count smoothing while minimizing memory usage.
   * `GPU_BATCHES` 2-3, 3 is probably better.
   * `WINDOW_BITS` 4-5, 4 for 256, 5 for 512

## Upgrades

  * [ ] Speeding up `run_overflow_coordinator_thread` work
     * `--cpu-fraction` at 2% wasn't saturating GPU testing.
       Increasing to 5% would help but would increase overflow work by 2-3X.
       Overflow work is probably sieved less agressievly than the main work so this is "less efficient".
       If the 5% of overflow takes 2x more prp tests, the overallwork is 105% which is great if it helps
       raise GPU utilization from 80% to 90%.

  * Consider for much later
    * Have `sieve_interval_cpu` do both directions, and do GPU offloading of `prev_prime`.
      * Pros:
        * Would free up 5+ CPU cores
      * Cons:
        * This is a fixed amount of work (doesn't change with `--cpu-fraction`)
    * Choose a consistent X to overflow at.
      * Pros:
        * If known before hand might simplify some of the CPU overflow sieve math & tracking
        * Can start sieves for next range ahead of time.
      * Cons:
        * Less dynamic flexibility

## TODO
  * [ ] Reduce the 10% of the time sieve is late
  * [ ] Tune `min-merit` math to understand cost of setting 28 vs 26.
  * [ ] Faster sieving
    * [ ] Tune 995'000 constant
    * Multithreading -> For small primes this is trivial -> For large primes it's also probably trivial
    * AVX512 scatter is maybe faster or not?
    * Why was my old Ryzen 3900x faster at sieving?
    * Efficency with current model is 10.6 PRP tests per `m`, at 100M this is 9.9 PRP/m (+7%)
      * Gain is probably more because initial sieve can be higher too.

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
