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
TODO trim
TODO cuda
$ sudo apt install libgmp10 libgmp-dev
$ sudo apt install mercurial build-essential automake autoconf bison make libtool texinfo m4
```

```
$ sudo apt install libmpfr-dev libmpc-dev libbenchmark-dev

$ python -m pip install --user gmpy2 primegapverify
```

```
$ git clone https://github.com/sethtroisi/prime-gap-gpu.git
$ cd prime-gap-gpu
```

## Misc

```
valgrind --suppressions=cuda.supp --leak-check=full ./gap_search_gpu -p 337 -d 2310 --mstart 10000000 --minc 200000 --max-prime 1 --min-merit 25 -v -v -v
```

## TODO

  * [ ] GPU batch probably doesn't need lock and can just wait on state.
  * [ ] Faster sieving
    * AVX
  * [ ] Offload some of overflow back to the GPU
    * Sieve near each M, have a list of next 20 X offsets
    * Can 95% reuse runner.run_test
    * Seems like only a 1-5% overhead on number of primes
      * Can increase "cpu"-fraction (and hence max-prime)
    * With less CPU work can do more sieving!
  * [ ] Do multiple threads for sieves at different X
  * [ ] Why does X=12 have twice as many unknowns at X=482?


## TODONE

  * [x] Wheel for divisors of d.
  * [x] Check a small percent of next_primes are actually prime
  * [x] Understand what sieve limit gmp is using for overflow
    * GMP is sieving to ~4M -> 10% less efficient
  * [x] Try changing vector<uint8_t> to vector<uint32_t>
    * This didn't seem to have any impact on speed, but I was told it might help in
      reduce mixed cache line access (8 vs 32 vs 64) control so I'll keep it.
