// Copyright 2026 Seth Troisi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gpu_sieve.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <gmp.h>
#include <omp.h>
#include <primesieve.hpp>

#include "gap_common.h"

using std::cout;
using std::endl;
using std::map;
using std::mutex;
using std::pair;
using std::vector;
using namespace std::chrono;

const uint32_t SEGMENT_THREADS = 1024;
// segment_kernel handles primes < SEGMENT_PRIME_MULT * segment_size
const uint32_t SEGMENT_PRIME_MULT = 4;

// Words of composite handled by each thread of unknown_*_kernel
const uint32_t UNKNOWN_CHUNK_WORDS = 32;
const uint32_t UNKNOWN_SCAN_THREADS = 1024;

// large_primes_batch_kernal estimates floor(X * neg_inv_K / p) with floats
const uint32_t BATCH_MAX_X = 1 << 17;

struct BatchX {
    uint32_t X[MAX_SIEVE_BATCH];
};

// support routines
inline void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0) {
  // check for cuda errors

  if(status!=cudaSuccess) {
    printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
    if(action!=NULL)
      printf("While running %s   (file %s, line %d)\n", action, file, line);
    exit(1);
  }
}
#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)


/** Index (mi >> 1) in [0, prime) of the first odd mi with (M_start + mi) * K + X = 0 mod prime. */
__device__ __forceinline__ uint32_t first_composite_index(
        const uint64_t M_start, const uint64_t X, const uint32_t p, const uint32_t neg_inv_K_p) {
    const int64_t prime = p;
    const int64_t neg_inv_K = neg_inv_K_p;

    // Negative for small M_start, requires prime be signed.
    int64_t neg_inner = M_start - X * neg_inv_K;
    int64_t temp = neg_inner % prime;
    int64_t mi_0 = temp <= 0 ? -temp : (prime - temp);
    assert( 0 <= mi_0 && mi_0 < prime );

    // Only odd mi are stored, requires K odd and M_start even.
    mi_0 += (mi_0 & 1) ? 0 : prime;
    return mi_0 >> 1;
}


/**
 * Each block sieves segment_size bits (as bytes in shared memory) with the smallest primes
 * then writes them, combined with the D wheel, to every word of its part of composite.
 */
__global__ void segment_kernel(
    const uint64_t M_start,
    const uint64_t X,
    const uint32_t segment_size,

    uint32_t __restrict__ *composite,

    const uint32_t wheel_size,
    const uint32_t __restrict__ *wheel_words,

    const uint32_t num_primes,
    const uint32_t __restrict__ *primes,
    const uint32_t __restrict__ *neg_inv_Ks    // -(K^-1) mod p
) {
    extern __shared__ uint32_t shared_words[];
    uint8_t *flags = (uint8_t*) shared_words;

    const uint32_t threads = blockDim.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t segment = blockIdx.x;

    for (uint32_t i = thread_idx; i < segment_size / 4; i += threads) {
        shared_words[i] = 0;
    }
    __syncthreads();

    const uint64_t segment_M_start = M_start + 2ull * segment * segment_size;
    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        const uint32_t prime = primes[pi];
        uint32_t t = first_composite_index(segment_M_start, X, prime, neg_inv_Ks[pi]);
        for (; t < segment_size; t += prime) {
            flags[t] = 1;
        }
    }
    __syncthreads();

    const uint32_t segment_words = segment_size / 32;
    const uint32_t word_start = segment * segment_words;
    for (uint32_t w = thread_idx; w < segment_words; w += threads) {
        // Multiply packs 8 bytes of 0/1 into 8 bits (byte i -> bit i)
        const uint64_t *f = (const uint64_t*) (flags + 32 * w);
        const uint64_t magic = 0x0102040810204080ull;
        uint32_t v;
        v  = (uint32_t) ((f[0] * magic) >> 56);
        v |= (uint32_t) ((f[1] * magic) >> 56) << 8;
        v |= (uint32_t) ((f[2] * magic) >> 56) << 16;
        v |= (uint32_t) ((f[3] * magic) >> 56) << 24;

        const uint32_t word = word_start + w;
        v |= wheel_words[(32 * word) % wheel_size];
        composite[word] = v;
    }
}

/** Primes with several multiples in the interval that are too large for segment_kernel. */
__global__ void medium_primes_kernal(
    const uint64_t M_start,
    const uint32_t M_INC_HALF,
    const uint64_t X,

    uint32_t __restrict__ *composite,

    const uint32_t num_primes,
    const uint32_t __restrict__ *primes,
    const uint32_t __restrict__ *neg_inv_Ks    // -(K^-1) mod p
) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        const uint32_t prime = primes[pi];
        uint32_t t = first_composite_index(M_start, X, prime, neg_inv_Ks[pi]);
        for (; t < M_INC_HALF; t += prime) {
            atomicOr(&composite[t >> 5], 1u << (t & 31));
        }
    }
}

/** Primes with at most one multiple in the interval, only used when X >= BATCH_MAX_X. */
__global__ void large_primes_kernal(
    const uint64_t M_start,
    const uint32_t M_INC_HALF,
    const uint64_t X,

    uint32_t __restrict__ *composite,

    const uint32_t num_primes,
    const uint32_t __restrict__ *primes,
    const uint32_t __restrict__ *neg_inv_Ks    // -(K^-1) mod p
) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        const uint32_t t = first_composite_index(M_start, X, primes[pi], neg_inv_Ks[pi]);
        if (t < M_INC_HALF) {
            atomicOr(&composite[t >> 5], 1u << (t & 31));
        }
    }
}

/**
 * Primes with at most one multiple in the interval for `batch` X at once,
 * reading the primes is the slow part so each X after the first is nearly free.
 */
__global__ void large_primes_batch_kernal(
    const uint64_t M_start,
    const uint32_t M_INC_HALF,
    const uint32_t batch,
    const BatchX Xs,

    // composite for Xs.X[i] starts at composite + i * composite_words
    uint32_t __restrict__ *composite,
    const uint32_t composite_words,

    const uint32_t num_primes,
    const uint32_t __restrict__ *primes,
    const uint32_t __restrict__ *neg_inv_Ks    // -(K^-1) mod p
) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        const uint32_t prime = primes[pi];
        const uint32_t neg_inv_K = neg_inv_Ks[pi];

        const uint32_t M_start_mod_p = M_start % prime;
        const float neg_inv_K_over_p = (float) neg_inv_K / (float) prime;

        for (uint32_t b = 0; b < batch; b++) {
            const uint32_t X = Xs.X[b];
            // Float error is < 0.04 for X < 2^17 so q is the true quotient or one less, r is in [0, 2 * prime)
            const uint32_t q = (int32_t) ((float) X * neg_inv_K_over_p - 0.05f);
            const uint64_t r = (uint64_t) X * neg_inv_K - (uint64_t) q * prime;
            assert( r < 2ull * prime );
            const uint32_t X_neg_inv_K = r >= prime ? r - prime : r;

            // (X * neg_inv_K - M_start) mod prime, everything from here stays below prime < 2^32
            const uint32_t mi_0 = X_neg_inv_K >= M_start_mod_p ?
                X_neg_inv_K - M_start_mod_p : X_neg_inv_K + (prime - M_start_mod_p);
            assert( mi_0 < prime );

            // Only odd mi are stored (mi_0 or mi_0 + prime), requires K odd and M_start even.
            const uint32_t t = (mi_0 >> 1) + ((mi_0 & 1) ? 0 : (prime >> 1));
            if (t < M_INC_HALF) {
                atomicOr(&composite[b * composite_words + (t >> 5)], 1u << (t & 31));
            }
        }
    }
}


/** Finalize part 1: count active and not composite bits in each chunk. */
__global__ void unknown_count_kernel(
    const uint32_t num_chunks,
    const uint32_t __restrict__ *composite,
    const uint32_t __restrict__ *active,
    uint32_t __restrict__ *chunk_counts
) {
    const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= num_chunks)
        return;

    const uint32_t *c = composite + chunk * UNKNOWN_CHUNK_WORDS;
    const uint32_t *a = active + chunk * UNKNOWN_CHUNK_WORDS;
    uint32_t count = 0;
    for (uint32_t i = 0; i < UNKNOWN_CHUNK_WORDS; i++) {
        count += __popc(a[i] & ~c[i]);
    }
    chunk_counts[chunk] = count;
}

/** Finalize part 2: counts -> exclusive prefix sum (in place) and total, run as a single block. */
__global__ void unknown_scan_kernel(
    const uint32_t num_chunks,
    uint32_t __restrict__ *chunk_counts,
    uint32_t __restrict__ *total
) {
    __shared__ uint32_t sums[UNKNOWN_SCAN_THREADS];
    assert( blockDim.x == UNKNOWN_SCAN_THREADS && gridDim.x == 1 );

    const uint32_t tid = threadIdx.x;
    const uint32_t per_thread = (num_chunks + UNKNOWN_SCAN_THREADS - 1) / UNKNOWN_SCAN_THREADS;
    const uint32_t start = min(tid * per_thread, num_chunks);
    const uint32_t end = min(start + per_thread, num_chunks);

    uint32_t mine = 0;
    for (uint32_t i = start; i < end; i++) {
        mine += chunk_counts[i];
    }
    sums[tid] = mine;
    __syncthreads();

    // Inclusive scan of sums (Hillis-Steele)
    for (uint32_t d = 1; d < UNKNOWN_SCAN_THREADS; d *= 2) {
        const uint32_t v = sums[tid] + (tid >= d ? sums[tid - d] : 0);
        __syncthreads();
        sums[tid] = v;
        __syncthreads();
    }

    uint32_t offset = sums[tid] - mine;
    for (uint32_t i = start; i < end; i++) {
        const uint32_t count = chunk_counts[i];
        chunk_counts[i] = offset;
        offset += count;
    }
    if (tid == UNKNOWN_SCAN_THREADS - 1) {
        *total = sums[tid];
    }
}

/** Finalize part 3: write mi = 2 * bit + 1 for every active and not composite bit (sorted). */
__global__ void unknown_write_kernel(
    const uint32_t num_chunks,
    const uint32_t __restrict__ *composite,
    const uint32_t __restrict__ *active,
    const uint32_t __restrict__ *chunk_offsets,
    const uint32_t capacity,
    uint32_t __restrict__ *unknowns
) {
    const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= num_chunks)
        return;

    const uint32_t *c = composite + chunk * UNKNOWN_CHUNK_WORDS;
    const uint32_t *a = active + chunk * UNKNOWN_CHUNK_WORDS;
    uint32_t offset = chunk_offsets[chunk];
    for (uint32_t i = 0; i < UNKNOWN_CHUNK_WORDS; i++) {
        uint32_t bits = a[i] & ~c[i];
        const uint32_t base = (chunk * UNKNOWN_CHUNK_WORDS + i) * 32;
        while (bits) {
            const uint32_t bit = __ffs(bits) - 1;
            bits &= bits - 1;
            // run_unknowns falls back to the CPU when there are more than capacity.
            if (offset < capacity) {
                unknowns[offset] = ((base + bit) << 1) | 1;
            }
            offset++;
        }
    }
}


/**
 * Return a^-1 mod p
 * a^-1 * a mod p = 1
 * assumes that gcd(a, p) = 1
 */
int64_t invert(int64_t a, int64_t p) {
    // Use extended euclidean algorithm to get
    // a * x + p * y = 1
    // inverse of a is then x
    int64_t x = 1, y = 0;
    int64_t x1 = 0, y1 = 1, a1 = a, p1 = p;
    while (p1) {
        int64_t q = a1 / p1;

        int64_t t = x1;
        x1 = x - q * x1;
        x = t;

        t = y1;
        y1 = y - q * y1;
        y = t;

        t = p1;
        p1 = a1 - q * p1;
        a1 = t;
    }
    return x < 0 ? (p + x) : x;
}


GPUSieve::GPUSieve(const struct Config& config) {
    // ----- Sieve stats & Merit Stuff
    init_K(config, K);
    const double K_log = _log(K);
    const double N_log = K_log + log(config.m_start);
    const double prob_prime = 1 / N_log - 1 / (N_log * N_log);

    this->verbose = config.verbose;
    this->profile = config.verbose >= 2;
    this->m_inc = config.m_inc;
    // SieveData / overflow stop at 10-11 * p, extra doesn't hurt.
    this->coprime_X = get_coprime_X(config, 20 * config.p);

    auto T0 = high_resolution_clock::now();

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&runner));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const uint64_t BITS = config.m_inc / 2;

    { // Compute prime stuff and copy over
        vector<uint32_t> host_primes;
        vector<uint32_t> host_neg_inv_Ks;

        // primes, inverses and all the kernel math are uint32_t
        assert( config.max_prime < (1ull << 32) );

        primesieve::iterator iter(3); // Ignore 2 which is weird
        uint64_t prime = iter.next_prime();
        assert( prime == 3 );
        for (; prime <= config.max_prime; prime = iter.next_prime()) {
            if (prime <= config.p && (config.d % prime > 0))
                continue;

            const uint64_t base_r = mpz_fdiv_ui(K, prime);
            const int64_t inv_K = invert(base_r, prime);
            if ((inv_K * base_r) % prime != 1) {
                printf("BAD INVERT: %lu -> %lu, %ld\n", prime, base_r, inv_K);
            }
            assert( (inv_K * base_r) % prime == 1 );
            const int64_t neg_inv_K = (prime - inv_K) % prime;
            assert( (neg_inv_K * base_r) % prime == (prime-1) );

            if (prime <= config.p) {
                assert( config.d % prime == 0 );
                d_neg_inv_K.emplace_back(prime, neg_inv_K);
                continue;
            }

            host_primes.push_back(prime);
            host_neg_inv_Ks.push_back(neg_inv_K);
        }
        num_primes = host_primes.size();

        auto count_below = [&](uint64_t limit) {
            return std::distance(host_primes.begin(),
                    std::lower_bound(host_primes.begin(), host_primes.end(), limit));
        };

        // Segments are bytes in shared memory.
        segment_size = 1 << 16;
        while (segment_size > prop.sharedMemPerBlockOptin)
            segment_size /= 2;
        assert(segment_size % (32 * SEGMENT_THREADS) == 0);
        CUDA_CHECK(cudaFuncSetAttribute(
            segment_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, segment_size));
        num_segments = (BITS + segment_size - 1) / segment_size;

        num_medium_primes = count_below(BITS);
        num_segment_primes = count_below(std::min<uint64_t>(SEGMENT_PRIME_MULT * segment_size, BITS));

        const size_t bytes = sizeof(uint32_t) * num_primes;
        CUDA_CHECK(cudaMallocAsync(&primes, bytes, runner));
        CUDA_CHECK(cudaMemcpyAsync(primes, host_primes.data(), bytes, cudaMemcpyHostToDevice, runner));

        CUDA_CHECK(cudaMallocAsync(&neg_inv_Ks, bytes, runner));
        CUDA_CHECK(cudaMemcpyAsync(neg_inv_Ks, host_neg_inv_Ks.data(), bytes, cudaMemcpyHostToDevice, runner));
    }

    // segment_kernel writes whole segments
    composite_bytes = (size_t) num_segments * segment_size / 8;
    const uint32_t composite_words = composite_bytes / sizeof(uint32_t);
    CUDA_CHECK(cudaMallocAsync(&composite, MAX_SIEVE_BATCH * composite_bytes, runner));

    // Batching is slower if the composites of a batch don't fit in L2 (T4), 8 is barely faster than 4 (L4).
    batch_size = std::clamp<size_t>(prop.l2CacheSize / (2 * composite_bytes), 1, 4);

    // Host only sees whole uint64, active_m_i_bits may have a few extra.
    copy_bytes = 8 * ((BITS + 63) / 64);
    assert(copy_bytes <= composite_bytes);
    CUDA_CHECK(cudaMallocHost((void**) &host_composite, copy_bytes + 24));
    CUDA_CHECK(cudaMallocHost((void**) &host_active, copy_bytes));
    memset(host_composite, 0, copy_bytes + 24);
    memset(host_active, 0, copy_bytes);

    { // GPU finalize, see run_unknowns
        assert(composite_words % UNKNOWN_CHUNK_WORDS == 0);
        num_chunks = composite_words / UNKNOWN_CHUNK_WORDS;
        // Normally < 6% of bits are unknown, run_unknowns handles more (slowly).
        unknown_capacity = std::max<uint32_t>(1024, BITS / 8);
        const size_t unknown_bytes = (size_t) MAX_SIEVE_BATCH * unknown_capacity * sizeof(uint32_t);

        CUDA_CHECK(cudaMallocAsync(&active, composite_bytes, runner));
        CUDA_CHECK(cudaMemsetAsync(active, 0, composite_bytes, runner));
        CUDA_CHECK(cudaMallocAsync(&chunk_offsets, sizeof(uint32_t) * num_chunks, runner));
        CUDA_CHECK(cudaMallocAsync(&unknowns, unknown_bytes, runner));
        CUDA_CHECK(cudaMallocAsync(&unknown_counts, MAX_SIEVE_BATCH * sizeof(uint32_t), runner));

        CUDA_CHECK(cudaMallocHost((void**) &host_unknowns, unknown_bytes));
        CUDA_CHECK(cudaMallocHost((void**) &host_unknown_counts, MAX_SIEVE_BATCH * sizeof(uint32_t)));
    }

    {
        D = config.d;
        K_mod_d = mpz_fdiv_ui(K, D);
        wheel_size = 1;
        for (const auto& [d, neg_inv_K] : this->d_neg_inv_K) {
            wheel_size *= d;
        }
        CUDA_CHECK(cudaMallocAsync(&wheel_words, MAX_SIEVE_BATCH * sizeof(uint32_t) * wheel_size, runner));
    }

    if (config.verbose >= 1) {
        setlocale(LC_NUMERIC, "");
        printf("\tGPUSieve(): %u segments of %u, primes: %'u segment, %'u medium, %'u large\n",
                num_segments, segment_size, num_segment_primes,
                num_medium_primes - num_segment_primes, num_primes - num_medium_primes);
        printf("\tGPUSieve(): L2 cache %d KB => batch size %u (%lu KB composite each)\n",
                prop.l2CacheSize / 1024, batch_size, composite_bytes / 1024);
        setlocale(LC_NUMERIC, "C");
    }

    cudaStreamSynchronize(runner);
    auto T1 = high_resolution_clock::now();
    auto gpu_setup_ms = duration_cast<milliseconds>(T1 - T0).count();
    printf("\tGPU<<<%d, %d>>> setup: %lu ms\n", GRID_SIZE, BLOCK_SIZE, gpu_setup_ms);
}

GPUSieve::~GPUSieve() {
    printf("GPUSieve Timings\n");
    printf("\ttotal sieving time: %.1f seconds / %lu sieves = %.2f ms / sieve\n",
            d_total, number_sieves, 1000 * d_total / number_sieves);
    printf("\tbatches: %lu (max batch size %u), %.2f sieves / batch\n",
            batches_run, batch_size, 1.0 * number_sieves / std::max<uint64_t>(1, batches_run));
    if (unknown_fallbacks)
        printf("\tMore than %u unknowns (slow CPU finalize) %lu times\n", unknown_capacity, unknown_fallbacks);
    if (!profile)
        printf("\t(kernels not timed individually, GPU time is included in finalize / copy)\n");
    const std::pair<const char*, double> timings[] = {
        {"wheel", d_wheel}, {"segment", d_segment}, {"medium", d_medium}, {"large", d_large},
        {"finalize", d_finalize}, {"copy", d_copy},
    };
    for (const auto& [name, seconds] : timings) {
        printf("\t%-8s: %7.3f seconds (%4.1f%%) %7.3f ms / sieve\n",
                name, seconds, 100.0 * seconds / d_total, 1000 * seconds / number_sieves);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(primes));
    CUDA_CHECK(cudaFree(neg_inv_Ks));
    CUDA_CHECK(cudaFree(composite));
    CUDA_CHECK(cudaFree(wheel_words));
    CUDA_CHECK(cudaFree(active));
    CUDA_CHECK(cudaFree(chunk_offsets));
    CUDA_CHECK(cudaFree(unknowns));
    CUDA_CHECK(cudaFree(unknown_counts));

    CUDA_CHECK(cudaFreeHost(host_composite));
    CUDA_CHECK(cudaFreeHost(host_active));
    CUDA_CHECK(cudaFreeHost(host_unknowns));
    CUDA_CHECK(cudaFreeHost(host_unknown_counts));
    CUDA_CHECK(cudaStreamDestroy(runner));

    mpz_clear(K);
}

void GPUSieve::set_batch_size(uint32_t batch) {
    assert(1 <= batch && batch <= MAX_SIEVE_BATCH);
    batch_size = batch;
    cache_X.clear();
}

void GPUSieve::set_active_bits(const uint64_t m_start, const uint64_t *bits, const size_t words) {
    assert(words * sizeof(uint64_t) >= copy_bytes);
    memcpy(host_active, bits, copy_bytes);
    host_active_m_start = m_start;
    host_active_pending = true;
}

int32_t GPUSieve::find_cached(
        const uint64_t m_start, const uint64_t X, const uint32_t max_p_i, bool need_unknowns) const {
    // A batch that used more primes than requested is fine.
    if (cache_m_start != m_start || std::min(this->num_primes, max_p_i) > cache_max_p_i)
        return -1;
    if (need_unknowns && !cache_has_unknowns)
        return -1;
    auto it = std::find(cache_X.begin(), cache_X.end(), X);
    return it == cache_X.end() ? -1 : std::distance(cache_X.begin(), it);
}

uint32_t GPUSieve::cached_or_sieve(
        const uint64_t m_start, const uint64_t m_inc,
        const uint64_t X, const uint32_t max_p_i, bool with_unknowns) {
    assert(m_inc == this->m_inc);
    assert(m_inc % 2 == 0);
    assert(m_start % 2 == 0);

    int32_t cache_i = find_cached(m_start, X, max_p_i, with_unknowns);
    if (cache_i < 0) {
        sieve_batch(m_start, X, max_p_i, with_unknowns);
        cache_i = 0;
    }
    assert(cache_X[cache_i] == X);
    return cache_i;
}

void GPUSieve::upload_wheels(const uint64_t m_start) {
    const uint32_t batch = cache_X.size();
    // One byte per bit of the wheel, with 32 extra so every 32 bit window can be read.
    vector<uint8_t> wheel_bits(wheel_size + 32);
    vector<uint32_t> host_wheel_words(batch * wheel_size);
    for (uint32_t b = 0; b < batch; b++) {
        const uint64_t X = cache_X[b];

        std::fill(wheel_bits.begin(), wheel_bits.end(), 0);
        for (const auto& [d, neg_inv_K] : this->d_neg_inv_K) {
            uint64_t mi_0 = (X * neg_inv_K + d - (m_start % d)) % d;
            mi_0 += (mi_0 & 1) ? 0 : d;
            assert( ((m_start + mi_0) * K_mod_d + X) % d == 0 );

            for (uint32_t i = (mi_0 >> 1) % d; i < wheel_bits.size(); i += d)
                wheel_bits[i] = 1;
        }

        // words[r] = bits [r, r + 32) of the wheel
        uint32_t *words = host_wheel_words.data() + b * wheel_size;
        uint32_t word = 0;
        for (uint32_t i = 0; i < 32; i++)
            word |= (uint32_t) wheel_bits[i] << i;
        for (uint32_t r = 0; r < wheel_size; r++) {
            words[r] = word;
            word = (word >> 1) | ((uint32_t) wheel_bits[r + 32] << 31);
        }
    }
    // Pageable memory, has been read (staged) when this returns.
    CUDA_CHECK(cudaMemcpyAsync(this->wheel_words, host_wheel_words.data(),
                sizeof(uint32_t) * host_wheel_words.size(), cudaMemcpyHostToDevice, runner));
}

void GPUSieve::sieve_batch(
        const uint64_t m_start, const uint64_t first_X, const uint32_t max_p_i, bool with_unknowns) {
    const uint32_t BITS = m_inc / 2;
    const uint32_t composite_words = composite_bytes / sizeof(uint32_t);

    // X and the X that are expected to be sieved next
    cache_X.clear();
    cache_X.push_back(first_X);
    for (auto it = std::upper_bound(coprime_X.begin(), coprime_X.end(), first_X);
            it != coprime_X.end() && cache_X.size() < batch_size; it++) {
        cache_X.push_back(*it);
    }
    cache_m_start = m_start;
    cache_max_p_i = std::min(this->num_primes, max_p_i);
    cache_has_unknowns = with_unknowns;
    batches_run += 1;

    const uint32_t batch = cache_X.size();
    const uint32_t segment_i = std::min(this->num_segment_primes, max_p_i);
    const uint32_t medium_i = std::min(this->num_medium_primes, max_p_i);
    const uint32_t large_i = std::min(this->num_primes, max_p_i);

    // Only syncs (so that the kernels before it are timed) when profiling.
    auto T = high_resolution_clock::now();
    auto add_time = [&](double &total) {
        if (profile) cudaStreamSynchronize(runner);
        auto now = high_resolution_clock::now();
        total += duration<double>(now - T).count();
        T = now;
    };

    upload_wheels(m_start);
    add_time(d_wheel);

    BatchX batch_X = {};
    bool use_batch_kernel = true;
    for (uint32_t b = 0; b < batch; b++) {
        const uint64_t X = cache_X[b];
        uint32_t *b_composite = this->composite + b * composite_words;
        batch_X.X[b] = X;
        use_batch_kernel &= X < BATCH_MAX_X;

        // Sets every word of composite, must be first.
        segment_kernel<<<num_segments, SEGMENT_THREADS, segment_size, runner>>>(
            m_start, X, segment_size,
            b_composite,
            this->wheel_size, this->wheel_words + b * wheel_size,
            segment_i,
            this->primes,
            this->neg_inv_Ks
        );
        add_time(d_segment);

        if (medium_i > segment_i) {
            medium_primes_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
                m_start, BITS, X,
                b_composite,
                medium_i - segment_i,
                this->primes + segment_i,
                this->neg_inv_Ks + segment_i
            );
        }
        add_time(d_medium);
    }

    if (large_i > medium_i && use_batch_kernel) {
        large_primes_batch_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
            m_start, BITS, batch, batch_X,
            this->composite, composite_words,
            large_i - medium_i,
            this->primes + medium_i,
            this->neg_inv_Ks + medium_i
        );
    } else if (large_i > medium_i) {
        for (uint32_t b = 0; b < batch; b++) {
            large_primes_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
                m_start, BITS, cache_X[b],
                this->composite + b * composite_words,
                large_i - medium_i,
                this->primes + medium_i,
                this->neg_inv_Ks + medium_i
            );
        }
    }
    add_time(d_large);

    if (!with_unknowns)
        return;

    assert(host_active_m_start == m_start);
    if (host_active_pending) {
        // Pinned memory, not modified till after the sync below.
        CUDA_CHECK(cudaMemcpyAsync(this->active, host_active, copy_bytes, cudaMemcpyHostToDevice, runner));
        host_active_pending = false;
    }

    const uint32_t blocks = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (uint32_t b = 0; b < batch; b++) {
        const uint32_t *b_composite = this->composite + b * composite_words;
        unknown_count_kernel<<<blocks, BLOCK_SIZE, 0, runner>>>(
                num_chunks, b_composite, this->active, this->chunk_offsets);
        unknown_scan_kernel<<<1, UNKNOWN_SCAN_THREADS, 0, runner>>>(
                num_chunks, this->chunk_offsets, this->unknown_counts + b);
        unknown_write_kernel<<<blocks, BLOCK_SIZE, 0, runner>>>(
                num_chunks, b_composite, this->active, this->chunk_offsets,
                unknown_capacity, this->unknowns + (size_t) b * unknown_capacity);
    }
    CUDA_CHECK(cudaMemcpyAsync(host_unknown_counts, this->unknown_counts, batch * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, runner));
    // Need the counts to know how much to copy.
    CUDA_CHECK(cudaStreamSynchronize(runner));

    // Copy the whole batch now so later run_unknowns() calls don't touch the GPU.
    for (uint32_t b = 0; b < batch; b++) {
        if (host_unknown_counts[b] > unknown_capacity)
            continue;
        const size_t offset = (size_t) b * unknown_capacity;
        CUDA_CHECK(cudaMemcpyAsync(host_unknowns + offset, this->unknowns + offset,
                    host_unknown_counts[b] * sizeof(uint32_t), cudaMemcpyDeviceToHost, runner));
    }
    CUDA_CHECK(cudaStreamSynchronize(runner));
    d_finalize += duration<double>(high_resolution_clock::now() - T).count();
}

const uint32_t* GPUSieve::run_unknowns(
        const uint64_t m_start, const uint64_t m_inc,
        const uint64_t X, const uint32_t max_p_i,
        uint32_t *count) {
    auto T0 = high_resolution_clock::now();

    const uint32_t cache_i = cached_or_sieve(m_start, m_inc, X, max_p_i, true);
    *count = host_unknown_counts[cache_i];

    const uint32_t *result = host_unknowns + (size_t) cache_i * unknown_capacity;
    if (*count > unknown_capacity) {
        // Too many to store on the GPU, copy composite and finalize here.
        unknown_fallbacks += 1;
        auto T1 = high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(host_composite, composite + cache_i * (composite_bytes / sizeof(uint32_t)),
                    copy_bytes, cudaMemcpyDeviceToHost, runner));
        CUDA_CHECK(cudaStreamSynchronize(runner));
        d_copy += duration<double>(high_resolution_clock::now() - T1).count();

        host_unknowns_fallback.clear();
        host_unknowns_fallback.reserve(*count);
        for (uint32_t i = 0; i < copy_bytes / sizeof(uint64_t); i++) {
            uint64_t bits = host_active[i] & ~host_composite[i];
            while (bits) {
                host_unknowns_fallback.push_back((((64 * i) + __builtin_ctzl(bits)) << 1) | 1);
                bits &= bits - 1;
            }
        }
        assert(host_unknowns_fallback.size() == *count);
        result = host_unknowns_fallback.data();
    }

    const double total_d = duration<double>(high_resolution_clock::now() - T0).count();
    number_sieves += 1;
    d_total += total_d;

    bool should_print = (
            (verbose + (X <= 12) >= 3)
            || (number_sieves < 1000 && (verbose + (number_sieves % 100 == 1) >= 2))
    );
    if (should_print) {
        printf("\tGPU sieve X=%lu %.4f seconds | %u/%lu unknown\n", X, total_d, *count, m_inc / 2);
    }
    return result;
}

uint64_t* GPUSieve::run(
        const uint64_t m_start, const uint64_t m_inc,
        const uint64_t X, const uint32_t max_p_i) {
    auto T0 = high_resolution_clock::now();

    const uint32_t cache_i = cached_or_sieve(m_start, m_inc, X, max_p_i, false);

    auto T1 = high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(host_composite, composite + cache_i * (composite_bytes / sizeof(uint32_t)),
                copy_bytes, cudaMemcpyDeviceToHost, runner));
    CUDA_CHECK(cudaStreamSynchronize(runner));
    auto T2 = high_resolution_clock::now();

    number_sieves += 1;
    d_total += duration<double>(T2 - T0).count();
    d_copy  += duration<double>(T2 - T1).count();

    return host_composite;
}
