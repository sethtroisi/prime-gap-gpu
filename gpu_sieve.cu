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

const uint32_t COMPRESS_BYTES_PER_THREAD = 256;

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


// __device__ uint32_t mod2310(uint32_t n) {
//     // magic constant from "Division by Invariant Integers using Multiplication"
//     const uint32_t magic_2310 = 1903916239;
//     const int32_t shift = 10;
//     // mult.hi.u32 = upper 32 bits of u32, u32 multiplication
//     uint32_t q;
//     asm("mul.hi.u32 %0, %1, %2;" : "=r"(q) : "r"(n), "r"(magic_2310));
//     q >>= shift;
//     //assert( q == (m_temp / 2310) );
//     return n - q * 2310;
// }


/** Called by host executed on device. */
__global__ void wheel_kernel(
    const uint32_t d_wheel_size,
    const uint8_t *d_wheel,
    const uint64_t composite_size,
    uint8_t *composite
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = global_idx; i < composite_size; i += stride) {
        uint32_t wheel_idx = i % d_wheel_size;
        composite[i] = d_wheel[wheel_idx];
    }
}

/** Called by host executed on device. */
__global__ void compress_kernel(
    const uint64_t composite_size,
    const uint8_t *composite,
    uint8_t *result
) {
    uint32_t threads = gridDim.x * blockDim.x;
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t size_per = COMPRESS_BYTES_PER_THREAD;
    assert( size_per % 8 == 0 );
    assert( composite_size % 8 == 0 ); // avoids tail loop from loop unrolling.

    uint32_t copies = (composite_size-1) / size_per + 1;
    assert(threads > copies);
    if (thread_idx >= copies)
        return;

    uint32_t start_i = thread_idx * size_per;
    uint32_t c_size = composite_size - start_i;

    uint32_t bytes = size_per < c_size ? size_per : c_size;
    assert( bytes % 8 == 0);

    uint32_t c_i = start_i;

    for (uint32_t j = 0; j < bytes; j += 8) {
        // 8x loop unrolled
        uint8_t t = (
            composite[c_i]
            | (composite[c_i+1] << 1)
            | (composite[c_i+2] << 2)
            | (composite[c_i+3] << 3)
            | (composite[c_i+4] << 4)
            | (composite[c_i+5] << 5)
            | (composite[c_i+6] << 6)
            | (composite[c_i+7] << 7)
        );

        result[c_i >> 3] = t;
        c_i += 8;
    }
}


/** Called by host executed on device. */
__global__ void small_primes_kernal(
    int64_t *thread_stats,

    /** config section **/
    const uint64_t global_M_start,
    const uint64_t global_M_INC_HALF,
    const uint32_t X,

    uint8_t *composite,

    uint32_t *primes,
    int32_t *neg_inv_Ks      // r^-1 mod p
) {
    // Indexing is hard for me
    // blockIdx.x / gridDim.x
    // threadIdx.x / blockDim.x
    assert( blockDim.x == BLOCK_SIZE );

    // For small primes everyone works on every prime at the sametime
    // this probably keeps threads in sync more

    uint32_t threads = BLOCK_SIZE;
    uint32_t thread_idx = threadIdx.x;

    // Each block handles 1 prime (makes everything have equal timing in the block)
    // Each thread handles a portion of M_INC_HALF
    // [global_M_start, global_M_INC_HALF) -> [thread_M_start, thread_M_inc)
    uint64_t thread_offset = global_M_INC_HALF * thread_idx / threads;
    uint64_t thread_offset_next = global_M_INC_HALF * (thread_idx + 1) / threads;
    uint64_t thread_M_start = global_M_start + (thread_offset << 1);

    uint32_t pi = blockIdx.x;
    const uint32_t prime = primes[pi];
    const int32_t neg_inv_K = neg_inv_Ks[pi];

    // -M_start % p
    int64_t mi_0_shift = prime - (thread_M_start % prime);

    // Safe from overflow as (SL * prime + prime) < int64
    int64_t mi_0 = (X * neg_inv_K + mi_0_shift) % prime;
    // benchmark as "? prime : 0" vs "* prime";
    //mi_0 += ((mi_0 & 1) == 0) * prime;
    mi_0 += (mi_0 & 1) ? 0 : prime;
    mi_0 >>= 1;

    // mi_0 is indexed from thread_M_start
    // Add thread_offset to correct for position in composite.
    for (uint32_t t = thread_offset + mi_0; t < thread_offset_next; t += prime) {
        composite[t] = true;
    }
}

/** Called by host executed on device. */
__global__ void medium_primes_kernal(
    int64_t *thread_stats,

    /** config section **/
    const uint64_t M_start,
    const uint32_t M_INC_HALF,
    const uint64_t X,

    uint8_t *composite,

    uint32_t num_primes,
    uint32_t *primes,
    int32_t *neg_inv_Ks      // r^-1 mod p
) {
    uint64_t t0 = clock64();
    uint32_t small_factors = 0;

    // Indexing is hard for me
    // blockIdx.x / gridDim.x
    // threadIdx.x / blockDim.x
    assert( gridDim.x == GRID_SIZE );
    assert( blockDim.x == BLOCK_SIZE );

    uint32_t threads = GRID_SIZE * BLOCK_SIZE;
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t pi_0 = thread_idx;
    for (uint32_t pi = pi_0; pi < num_primes; pi += threads) {
        const uint64_t prime = primes[pi];
        const int64_t neg_inv_K = neg_inv_Ks[pi];

        // -M_start % p
        int64_t mi_0_shift = prime - (M_start % prime);
        {
            // Safe from overflow as (SL * prime + prime) < int64
            int64_t mi_0 = (X * neg_inv_K + mi_0_shift) % prime;
            // benchmark as "? prime : 0" vs "* prime";
            //mi_0 += ((mi_0 & 1) == 0) * prime;
            mi_0 += (mi_0 & 1) ? 0 : prime;
            mi_0 >>= 1;

            for (uint32_t t = mi_0; t < M_INC_HALF; t += prime) {
                composite[t] = true;
                small_factors += 1;
            }
        }
    }

    uint64_t t1 = clock64();

    // 4 is stats_per_thread.
    int index = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    thread_stats[4 * index + 0] = t0;
    thread_stats[4 * index + 1] = t1;
    thread_stats[4 * index + 2] = small_factors;
    thread_stats[4 * index + 3] = index;
}

/** Called by host executed on device. */
__global__ void large_primes_kernal(
    int64_t *thread_stats,

    /** config section **/
    const uint64_t M_start,
    const uint32_t M_INC_HALF,
    const uint64_t X,

    uint8_t *composite,

    uint32_t num_primes,
    uint32_t *primes,
    int32_t *neg_inv_Ks      // r^-1 mod p
) {
    uint64_t t0 = clock64();
    uint32_t small_factors = 0;

    // Indexing is hard for me
    // blockIdx.x / gridDim.x
    // threadIdx.x / blockDim.x
    assert( gridDim.x == GRID_SIZE );
    assert( blockDim.x == BLOCK_SIZE );

    uint32_t threads = GRID_SIZE * BLOCK_SIZE;
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t pi_0 = thread_idx;
    for (uint32_t pi = pi_0; pi < num_primes; pi += threads) {
        const uint64_t prime = primes[pi];
        const int64_t neg_inv_K = neg_inv_Ks[pi];

        // -M_start % p
        int64_t mi_0_shift = prime - (M_start % prime);

        // Safe from overflow as (SL * prime + prime) < int64
        int64_t mi_0 = (X * neg_inv_K + mi_0_shift) % prime;
        // benchmark as "? prime : 0" vs "* prime";
        //mi_0 += ((mi_0 & 1) == 0) * prime;
        mi_0 += (mi_0 & 1) ? 0 : prime;
        mi_0 >>= 1;

        /**
         * Potentially optimization to uncondiontally set mi_0 or M_INC_HALF
         * Didn't change benchmarking this is also only 7-20% of execution time
         */
        if (mi_0 < M_INC_HALF) {
            composite[mi_0] = true;
            small_factors += 1;
        }
    }

    uint64_t t1 = clock64();

    // 4 is stats_per_thread.
    int index = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
    thread_stats[4 * index + 0] = t0;
    thread_stats[4 * index + 1] = t1;
    thread_stats[4 * index + 2] = small_factors;
    thread_stats[4 * index + 3] = index;
}



/**
 * Return a^-1 mod p
 * a^-1 * a mod p = 1
 * assumes that gcd(a, p) = 1
 */
int32_t invert(int32_t a, int32_t p) {
    // Use extended euclidean algorithm to get
    // a * x + p * y = 1
    // inverse of a is then x
    int x = 1, y = 0;
    int x1 = 0, y1 = 1, a1 = a, p1 = p;
    while (p1) {
        int32_t q = a1 / p1;

        int32_t t = x1;
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

    // ----- Allocate memory

    this->verbose = config.verbose;

    { // GPU Setup part
        auto T0 = high_resolution_clock::now();

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&runner));

        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

        { // Compute prime stuff and copy over
            vector<uint32_t> host_primes;
            vector<int32_t> host_neg_inv_Ks;

            primesieve::iterator iter(3); // Ignore 2 which is weird
            uint32_t prime = iter.next_prime();
            assert( prime == 3 );
            num_primes = 0;
            for (; prime <= config.max_prime; prime = iter.next_prime()) {
                if (prime <= config.p && (config.d % prime > 0))
                    continue;

                const uint64_t base_r = mpz_fdiv_ui(K, prime);
                const int32_t inv_K = invert(base_r, prime);
                if ((inv_K * base_r) % prime != 1) {
                    printf("BAD INVERT: %u -> %lu, %d\n", prime, base_r, inv_K);
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
                num_primes += 1;
            }
            //printf("Processed %u primes\n", num_primes);
            num_medium_primes = std::distance(host_primes.begin(),
                    std::lower_bound(host_primes.begin(), host_primes.end(), config.m_inc / 2));
            assert(2 * host_primes[num_medium_primes-1] < config.m_inc);
            assert(2 * host_primes[num_medium_primes] >= config.m_inc);

            const size_t bytes = sizeof(uint32_t) * num_primes;
            CUDA_CHECK(cudaMallocAsync(&primes, bytes, runner));
            CUDA_CHECK(cudaMemcpyAsync(primes, host_primes.data(), bytes, cudaMemcpyHostToDevice, runner));

            CUDA_CHECK(cudaMallocAsync(&neg_inv_Ks, bytes, runner));
            CUDA_CHECK(cudaMemcpyAsync(neg_inv_Ks, host_neg_inv_Ks.data(), bytes, cudaMemcpyHostToDevice, runner));
            // After reading https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
            // I believe memcpyAsync is only async for GPU and CPU is sync with respect to the host.
        }

        composite_bytes = sizeof(char) * config.m_inc / 2 + 2;
        CUDA_CHECK(cudaMallocAsync(&composite, composite_bytes, runner));
        CUDA_CHECK(cudaMallocAsync(&composite_compressed, composite_bytes / 8 + 1, runner));

        {
            CUDA_CHECK(cudaMallocHost((void**) &host_thread_stats, thread_stats_bytes));
            CUDA_CHECK(cudaMallocAsync(&thread_stats, thread_stats_bytes, runner));
            CUDA_CHECK(cudaMemsetAsync(thread_stats, 0, thread_stats_bytes, runner));
        }

        host_composite_bytes = composite_bytes;
        cudaMallocHost((void**) &host_composite, host_composite_bytes);

        {
            D = config.d;
            K_mod_d = mpz_fdiv_ui(K, D);
            d_wheel_bytes = (4*D);
            if (GRID_SIZE * d_wheel_bytes > config.m_inc) {
                printf("\td_wheel_bits only tiles a few times!\n");
            }
            CUDA_CHECK(cudaMallocAsync(&D_wheel, d_wheel_bytes, runner));

            host_wheel.resize(d_wheel_bytes, 0);
        }

        if (config.verbose >= 1) {
            setlocale(LC_NUMERIC, "");
            printf("\tGPUSieve(): malloced: primes: %'d, composite: %lu MB + %lu MB\n",
                    num_primes, composite_bytes / 1024 / 1024, host_composite_bytes / 1024 / 1024);
            setlocale(LC_NUMERIC, "C");
        }

        cudaStreamSynchronize(runner);
        auto T1 = high_resolution_clock::now();
        auto gpu_setup_ms = duration_cast<milliseconds>(T1 - T0).count();
        printf("\tGPU<<<%d, %d>>> setup: %lu ms\n", GRID_SIZE, BLOCK_SIZE, gpu_setup_ms);
    }
}

GPUSieve::~GPUSieve() {
    printf("GPUSieve Timings\n");
    printf("\ttotal sieving time: %.1f seconds / %lu sieves = %.1f ms / sieve\n",
            d_total, number_sieves, 1000 * d_total / number_sieves);
    printf("\twheel1 : %5.2f seconds (%4.1f%%)\n", d_w1, 100.0 * d_w1 / d_total);
    printf("\twheel2 : %5.2f seconds (%4.1f%%)\n", d_w2, 100.0 * d_w2 / d_total);
    printf("\tsmall  : %5.2f seconds (%4.1f%%)\n", d_k1, 100.0 * d_k1 / d_total);
    printf("\tmedium : %5.2f seconds (%4.1f%%)\n", d_k2, 100.0 * d_k2 / d_total);
    printf("\tlarge  : %5.2f seconds (%4.1f%%)\n", d_k3, 100.0 * d_k3 / d_total);
    printf("\tcopy   : %5.2f seconds (%4.1f%%)\n", d_copy, 100.0 * d_copy / d_total);
    printf("\n");

    CUDA_CHECK(cudaFreeHost(host_thread_stats));
    CUDA_CHECK(cudaFree(thread_stats));

    CUDA_CHECK(cudaFree(composite));

    CUDA_CHECK(cudaFree(primes));
    CUDA_CHECK(cudaFree(neg_inv_Ks));

    CUDA_CHECK(cudaFreeHost(host_composite));
    CUDA_CHECK(cudaStreamDestroy(runner));

    mpz_clear(K);
}

uint8_t* GPUSieve::run(
        const uint64_t m_start, const uint64_t m_inc,
        const uint64_t X, const uint32_t max_p_i) {

    assert(m_inc % 2 == 0);
    uint32_t BITS = m_inc / 2;
    double w1_d, w2_d, small_d, medium_d, large_d, total_d;

    auto t_sieve_start = high_resolution_clock::now();
    { // Run GPU Sieve!
        std::fill(host_wheel.begin(), host_wheel.end(), 0);
        for (const auto& [d, neg_inv_K] : this->d_neg_inv_K) {
            if (d == 2) continue;

            uint64_t mi_0 = (X * neg_inv_K + d - (m_start % d)) % d;
            mi_0 += (mi_0 & 1) ? 0 : d;
            assert( ((m_start + mi_0) * K_mod_d + X) % d == 0 );

            for (uint32_t i = (mi_0 >> 1); i < d_wheel_bytes; i += d)
                host_wheel[i] = 1;
        }
        CUDA_CHECK(cudaMemcpyAsync(this->D_wheel, host_wheel.data(), d_wheel_bytes, cudaMemcpyHostToDevice, runner));
        cudaStreamSynchronize(runner);
        auto W1 = high_resolution_clock::now();

        wheel_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
            this->d_wheel_bytes,
            this->D_wheel,
            BITS,
            this->composite
        );
        cudaStreamSynchronize(runner);

        auto T1 = high_resolution_clock::now();

        // Should be a small multiple of GRID_SIZE
        num_small_primes = 2 * GRID_SIZE;
        assert( num_small_primes < this->num_primes);
        small_primes_kernal<<<num_small_primes, BLOCK_SIZE, 0, runner>>>(
            this->thread_stats,
            m_start, BITS, X,
            this->composite,
            this->primes,
            this->neg_inv_Ks
        );

        cudaStreamSynchronize(runner);
        auto T2 = high_resolution_clock::now();

        uint32_t medium_i = std::min(this->num_medium_primes, max_p_i);
        assert( medium_i > this->num_small_primes );
        medium_primes_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
            this->thread_stats,
            m_start, BITS, X,
            this->composite,
            medium_i - this->num_small_primes,
            this->primes + this->num_small_primes,
            this->neg_inv_Ks + this->num_small_primes
        );

        // TODO check consequence of sync here.
        cudaStreamSynchronize(runner);
        auto T3 = high_resolution_clock::now();

        uint32_t large_i = std::min(this->num_primes, max_p_i);
        if (large_i > medium_i) {
            // Could be overlapped with medium_primes.
            large_primes_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
                this->thread_stats,
                m_start, BITS, X,
                this->composite,
                large_i - medium_i,
                this->primes + medium_i,
                this->neg_inv_Ks + medium_i
            );
        }

        //printf("Running small to %u, medium to %u, large to %u\n",
        //        this->num_small_primes-1, medium_i-1, large_i-1);

        cudaStreamSynchronize(runner);
        auto T4 = high_resolution_clock::now();

        w1_d  = duration<double>(W1 - t_sieve_start).count();
        w2_d  = duration<double>(T1 - W1).count();
        small_d  = duration<double>(T2 - T1).count();
        medium_d = duration<double>(T3 - T2).count();
        large_d  = duration<double>(T4 - T3).count();
    }

    if (0) { // Read thread stats
        CUDA_CHECK(cudaMemcpyAsync(host_thread_stats, thread_stats, thread_stats_bytes,
                   cudaMemcpyDeviceToHost, runner));
        cudaStreamSynchronize(runner);
        auto first_t0 = host_thread_stats[0];
        for (size_t ti = 0; ti < GRID_SIZE * BLOCK_SIZE; ti++) {
            first_t0 = std::min(first_t0, host_thread_stats[stats_per_thread * ti + 0]);
        }

        for (size_t ti = 0; ti < GRID_SIZE * BLOCK_SIZE; ti++) {
            const auto s = host_thread_stats + (stats_per_thread * ti);
            //auto t0 = s[0];
            //auto t1 = s[1];
            //auto small_factors = s[2];
            auto verify = s[3];
            //if (ti % 173 == 0) {
            //    printf("\tt%-5lu | t0 offset = %-13ld | t1-t0 = %-12ld | factors: %ld\n",
            //            ti, t0 - first_t0, t1 - t0, small_factors);
            //}
            assert(verify == ti);
        }
    }

    if (1) { // Parse results back to composite.
        auto T0 = high_resolution_clock::now();

        // Compress byte indexed data to bit indexed data to save on transfer cost.
        {
            uint32_t intervals = (BITS - 1) / COMPRESS_BYTES_PER_THREAD + 1;
            uint32_t block_size = 4 * BLOCK_SIZE;
            uint32_t needed_blocks = (intervals - 1) / (2*BLOCK_SIZE) + 1;
            //printf("\tcompress_kernel<<<%u, %u>>>\n", needed_blocks, 2*BLOCK_SIZE);
            compress_kernel<<<needed_blocks, block_size, 0, runner>>>(
                    BITS, composite, composite_compressed);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_composite, composite_compressed, composite_bytes/8,
                   cudaMemcpyDeviceToHost, runner));
        cudaStreamSynchronize(runner);

        auto T1 = high_resolution_clock::now();

        auto copy_d = duration<double>(T1 - T0).count();
        total_d = duration<double>(T1 - t_sieve_start).count();

        number_sieves += 1;
        d_total += total_d;
        d_w1    += w1_d;
        d_w2    += w2_d;
        d_k1    += small_d;
        d_k2    += medium_d;
        d_k3    += large_d;
        d_copy  += copy_d;

        bool should_print = (
                (verbose + (X <= 12) >= 3)
                || (number_sieves < 1000 && (verbose + (number_sieves % 100 == 1) >= 2))
        );
        if (should_print) {
            // running this code takes a 50ms?!?
            uint32_t num_composite = 0;
            for (uint32_t mi = 0; mi < (BITS + 7) / 8; mi++) {
                num_composite += __builtin_popcount(host_composite[mi]);
            }
            printf("\tGPU sieve %.3f | wheel: %.4f, %.4f, kernels: %.4f, %.4f, %.4f, copy-back: %.4f  "
                    "| %u/%u composite\n",
                    total_d, w1_d, w2_d, small_d, medium_d, large_d, copy_d, num_composite, BITS);
        }
    }

    return host_composite;
}
