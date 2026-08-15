# Table of Contents

- [Benchmarks](#benchmarks)

# Benchmarks

Aug 15

```
./gap_search_gpu -p 151 -d 2310 --mstart 683900000000 --minc 120000000 --max-prime 60 --min-merit 26.5 --cpu-fraction 0.020 --cpu-threads 6 -v

GPU Timings:
	m processed    : 249,350,649 (1,307,698/sec)
	total tests    : 1,236,492,173 (19.8% prime) (6,484,678/sec)
	total batches   : 302,600 (1586.96003 secs/batch)
	waits on no active_m(5ms) : 475
	waits on no next_tests(1ms): 23031
	filling batches: 46.2 seconds (24.2%)
	waiting filled : 3.4 seconds (1.8%)
	running on gpu : 304.9 seconds (159.9%)
	waiting done   : 0.0 seconds (0.0%)
	results        : 61.2 seconds (32.1%)
	batch fill %   : 99.8% (% fill), 0.5% (% partial batch)

SIEVE Timings:
	total_m: 1,200,000,000 (6,328,832/second) 189.6 seconds
	sieves: 1430 (0.0% early exit)
	avg prime: 29,969,929
	finalize_time(17.1%): 22.6 seconds (0.016/sieve)
	total_time: 131.8 seconds (0.092/sieve)
	total_active: 9,749,360,083, total_unknown: 1,333,110,835 (13.67%)
	active / run: 6,817,734, unknown / run: 932,245


```


Aug 12?

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
	total_m: 100,000,000 (510,850/second) 195.8 seconds
	sieves: 1308
	finalize_time(6.6%): 4.8 seconds (0.004/sieve)
	total_time: 73.2 seconds (0.056/sieve)
	total_active: 2,238,382,250, total_unknown: 362,048,860 (16.2%)
	active / run: 1,711,301, unknown / run: 276,795
```
