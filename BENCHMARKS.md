# Table of Contents

- [Benchmarks](#benchmarks)

# Benchmarks

## Aug 17

This has five records in 2B ~5-6minutes
```
make BITS=256 gap_search_gpu && ./gap_search_gpu -p 151 -d 2310 --mstart 956702000000 --minc 120000000 --max-prime 60 --min-merit 26.5 --cpu-fraction 0.020 --cpu-threads 6 -v
```


## Aug 15

```
./gap_search_gpu -p 151 -d 2310 --mstart 683900000000 --minc 120000000 --max-prime 60 --min-merit 26.5 --cpu-fraction 0.020 --cpu-threads 6 -v

GPU Timings:
	m processed    : 249,350,649 (1,395,644/sec)
	total tests    : 1,237,179,447 (19.8% prime) (6,924,638/sec)
	total batches   : 302,766 (1694.61672 secs/batch)
	waits on no active_m(5ms) : 476
	waits on no next_tests(1ms): 18831
	filling batches: 44.9 seconds (25.1%)
	waiting filled : 3.4 seconds (1.9%)
	running on gpu : 284.2 seconds (159.1%)
	waiting done   : 0.0 seconds (0.0%)
	results        : 3.5 seconds (2.0%)
	batch fill %   : 99.8% (% fill), 0.5% (% partial batch)

SIEVE Timings:
	total_m: 1,200,000,000 (6,758,578/second) 177.6 seconds
	sieves: 1430 (0.0% early exit)
	avg prime: 29,684,490
	finalize_time(18.3%): 23.2 seconds (0.016/sieve)
	total_time: 126.7 seconds (0.089/sieve)
	total_active: 9,739,073,086, total_unknown: 1,332,498,357 (13.68%)
	active / run: 6,810,540, unknown / run: 931,817
```

```
time ./gap_search_gpu -p 337 -d 2310 --mstart 81580000000 --minc 40000000 --max-prime 100 --min-merit 26 --cpu-fraction 0.013 --cpu-threads 7

GPU Timings:
	m processed    : 41,558,442 (186,514/sec)
	total tests    : 452,825,697 (9.1% prime) (2,032,284/sec)
	total batches   : 111,486 (500.34986 secs/batch)
	waits on no active_m(5ms) : 270
	waits on no next_tests(1ms): 147
	filling batches: 14.8 seconds (6.6%)
	waiting filled : 1.1 seconds (0.5%)
	running on gpu : 402.4 seconds (180.6%)
	waiting done   : 0.0 seconds (0.0%)
	results        : 0.8 seconds (0.4%)
	batch fill %   : 99.2% (% fill), 1.7% (% partial batch)


SIEVE Timings:
	total_m: 200,000,000 (899,233/second) 222.4 seconds
	sieves: 1871 (0.0% early exit)
	avg prime: 49,963,193
	finalize_time(8.5%): 7.4 seconds (0.004/sieve)
	total_time: 87.3 seconds (0.047/sieve)
	total_active: 3,060,682,870, total_unknown: 475,876,055 (15.55%)
	active / run: 1,635,854, unknown / run: 254,343
```


## Aug 12?

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
