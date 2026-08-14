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

```
time ./gap_search_gpu -p 337 -d 2310 --mstart 81580000000 --minc 40000000 --max-prime 36 --min-merit 26 --cpu-fraction 0.013 --cpu-threads 7

GPU Timings:
	m processed    : 33,246,754 (169,525/sec)
	total tests    : 355,364,613 (9.2% prime) (1,812,012/sec)
	waits on no active_m(10ms) : 88
	waits on no next_tests(2ms): 11214
	filling batches: 11.0 seconds (5.6%)
	waiting filled : 0.8 seconds (0.4%)
	running on gpu : 314.9 seconds (160.6%)
	waiting done   : 4.3 seconds (2.2%)
	results        : 7.9 seconds (4.0%)
	batch fill %   : 99.3% (% fill), 1.5% (% partial batch)


SIEVE Timings:
	total_m: 200,000,000 (1,021,700/second) 195.8 seconds
	sieves: 1308
	finalize_time(6.6%): 4.8 seconds (0.004/sieve)
	total_time: 73.2 seconds (0.056/sieve)
	total_active: 2,238,382,250, total_unknown: 362,048,860 (16.2%)
	active / run: 1,711,301, unknown / run: 276,795
```

## TODO

  * [ ] Faster sieving
    * AVX
    * faster finalize would help with early exit
  * [ ] Try changing vector<uint8_t> to vector<uint32_t>
  * [ ] Why does X=12 have twice as many unknowns at X=482?

## TODONE

  * [x] Wheel for divisors of d.
  * [x] Check a small percent of next_primes are actually prime
  * [x] Understand what sieve limit gmp is using for overflow
    * GMP is sieving to ~4M -> 10% less efficient
