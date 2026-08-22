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

  * `--cpu-fraction`
    * Increasing leads to less sparse sieves across X, trades of for more overflow work
  * `OVERFLOW_SIEVE_LIMIT`: TODO
    * Trades CPU sieving for GPU time, look at `total time     : sieve` from `CPU OVERFLOW Timing`
  * `max-prime` better to increase at some point top primes never run
  * `minc`: TODO add some metric to tune on.

These likely are set good enough

   * `OPEN_SIEVES` 3-6 is probably great balance of enough unknown count smoothing while minimizing memory usage.
   * `GPU_BATCHES` 2-3, 3 is probably better.


## TODO

  * [ ] Faster sieving
    * [ ] Multithreading -> For small primes this is trivial -> For large primes it's also probably trivial
    * AVX?
    * Why was a Ryzen 3900x 3x faster at sieving?
  * [ ] Offload some of overflow back to the GPU
    * Sieve near each M, have a list of next 20 X offsets
    * Reuse `runner.run_test` and `sieve_interval_cpu`
    * Seems like only a 1-5% overhead on number of primes
      * Can increase "cpu"-fraction (and hence max-prime)
    * Instead of sieving 500 m for 1 active m, sieve 5x more numbers than needed
      * Converts a lot of L3 access to L1/L2 access.

  * [ ] Investigate GAP MISMATCH reporting, I think some of the variables are being reused
  * [ ] Why does X=12 have twice as many unknowns at X=482?


## TODONE

  * [x] Wheel for divisors of d.
  * [x] Check a small percent of next_primes are actually prime
  * [x] Understand what sieve limit gmp is using for overflow
    * GMP is sieving to ~4M -> 10% less efficient
  * [x] Try changing vector<uint8_t> to vector<uint32_t>
    * This didn't seem to have any impact on speed, but I was told it might help in
      reduce mixed cache line access (8 vs 32 vs 64) control so I'll keep it.
