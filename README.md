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
time ./gap_search_gpu -p 337 -d 2310 --mstart 29100000000 --minc 8000000 --max-prime 32 --min-merit 26 --cpu-fraction 0.011 --cpu-threads 7

GPU Timings:
	waits on no active_m(10ms) : 4179
	waits on no next_tests(2ms): 2968610
	filling batches: 767.0 seconds (4.4%)
	waiting filled : 60.4 seconds (0.3%)
	running on gpu : 19891.1 seconds (113.8%)
	waiting done   : 220.4 seconds (1.3%)
	results        : 696.1 seconds (4.0%)
	Batch fill %   : 96.3% (% fill), 7.5% (% partial batch)


SIEVE Timings:
	total_m: 9944000000 (569117/second) 17472.7 seconds
	sieves: 420964
	finalize_time(2.5%): 286.8 seconds (0.001/sieve)
	total_time: 11563.1 seconds (0.027/sieve)
	total_active: 138842455702, total_unknown: 22596117891 (16.3%)
	active / run: 329820, unknown / run: 53677

	Finalizing(stage 2): 16089 open, 22210000 processed
	Finalizing(stage 2): 11089 open, 22215000 processed
	Finalizing(stage 2): 6089 open, 22220000 processed
	Finalizing(stage 2): 1089 open, 22225000 processed

CPU OVERFLOW Timing:
	total tested: 22226089
	next prime only: 3761386, both sides: 18464699
	> 26.0 merit: 80 (24 = 30.0% bad next_prime)
```

## TODO

 * [ ] `save_partial_at_exit` save offset (and what merit this represents) + currently active M.
 * [ ] Might have missed '927598685 * 337# / 210 - 4538'?
