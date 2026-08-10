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

#include "sieve_small.h"

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


// BIT_IS_BIT means 8x less GPU memory, faster transfers but, slightly slower compute
// A very small percentage of factors get lost.
// #define BIT_IS_BIT

// support routines
void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0) {
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
__global__ void method2_medium_primes_kernal(
    int64_t *thread_stats,

    /** config section **/
    const uint64_t M_start,
    const uint32_t M_inc,
    const uint32_t X,

    uint8_t *composite,

    uint32_t num_primes,
    uint32_t *primes,
    // uint32_t *remainders, // r = K mod p
    int32_t *neg_inv_Ks      // r^-1 mod p
) {
    uint64_t t0 = clock64();

    // Indexing is hard for me
    // blockIdx.x / gridDim.x
    // threadIdx.x / blockDim.x
    assert( gridDim.x == GRID_SIZE );
    assert( blockDim.x == BLOCK_SIZE );

    // TODO with block/thread splitting pi/coprime backwards
    // uint32_t print_mult = 10000;
    // uint32_t next_print = print_mult;
    // uint32_t next_mult = 5 * print_mult;

    uint32_t small_factors = 10;

    assert( BLOCK_SIZE == 64 );
    uint32_t pi_0 = (blockIdx.x << 1) + (threadIdx.x & 1);


    for (uint32_t pi = pi_0; pi < num_primes; pi += GRID_SIZE) {
    //for (uint32_t pi = threadIdx.x; pi < num_primes; pi += BLOCK_SIZE) {
        const uint32_t prime = primes[pi];
        //const uint32_t base_r = remainders[pi];
        const int32_t neg_inv_K = neg_inv_Ks[pi];

        // {
        //     // as large as prime^2
        //     uint64_t t = ((uint64_t) neg_inv_K) * base_r;
        //     if (index == 0 && prime >= next_print) {
        //         printf("\tGPU @ prime(%u): %u\n", pi, prime);

        //         if (next_print == next_mult) {
        //             print_mult *= 10;
        //             next_print = print_mult;
        //             next_mult = 5 * print_mult;
        //         } else {
        //             next_print += print_mult;
        //         }
        //     }
        //     assert(t % prime == (prime-1));
        //     assert(base_r < prime);
        //     assert(0 < neg_inv_K);
        //     assert(neg_inv_K < prime);
        // }

        // -M_start % p
        int64_t mi_0_shift = prime - (M_start % prime);
        if (mi_0_shift == prime) {
            mi_0_shift = 0;
        }

        const uint8_t M_parity_check = M_start & 1;
        uint32_t shift = prime << 1;

        {
            // Safe from overflow as (SL * prime + prime) < int64
            int64_t mi_0 = (X * neg_inv_K + mi_0_shift) % prime;
            // benchmark as "? prime : 0" vs "* prime";
            mi_0 += (((X ^ mi_0) & 1) == M_parity_check) * prime;

            uint32_t mi = mi_0;
            for (; mi < M_inc; mi += shift) {
                // TODO if (...) continue code only makes GPU 10% faster
                // Maybe replace with is_m_coprime will be a better check, given fast memory.

                // After initial value this increases by (shift * K_mod2310) % 2310
                //uint32_t n_mod2310 = ((K_mod2310 * m_mod2310) + X) % 2310;
                //uint32_t n_mod2310 = mod2310((K_mod2310 * m_mod2310) + X);

#ifdef BIT_IS_BIT
                // TODO 1<<(index&7) this could be extracted to the top to save a few operations
                composite[mi >> 3] |= 1 << (mi & 7);
#else
                composite[mi] = true;
#endif  // BIT_IS_BIT
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
    const double K_log = prob_prime_and_stats(config, K);
    const double N_log = K_log + log(config.m_start);
    const double prob_prime = 1 / N_log - 1 / (N_log * N_log);

    // ----- Allocate memory

    // Various pre-calculated arrays of is_coprime arrays

    { // Output setup
        // TODO any other variables?
        M_start       = config.m_start;
        //K_mod2310 = caches.K_mod2310;
        // Only one copy of these two
        {
        //    m_inc.swap(caches.valid_mi);
        //    host_reindex.swap(caches.m_reindex);
        }
    }

    { // GPU Setup part
        auto T0 = high_resolution_clock::now();

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&runner));

        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

        { // Compute prime stuff and copy over
            vector<uint32_t> host_primes;
            // vector<uint32_t> host_remainders;
            vector<int32_t> host_neg_inv_Ks;

            primesieve::iterator iter(3); // Ignore 2 which is weird
            uint32_t prime = iter.next_prime();
            assert( prime == 3 );
            num_primes = 0;
            for (; prime <= config.max_prime; prime = iter.next_prime()) {
                if (prime <= config.p && config.d % prime != 0) {
                    continue;
                }

                const uint64_t base_r = mpz_fdiv_ui(K, prime);
                const int32_t inv_K = invert(base_r, prime);
                if ((inv_K * base_r) % prime != 1) {
                    printf("BAD INVERT: %u -> %lu, %d\n", prime, base_r, inv_K);
                }
                assert( (inv_K * base_r) % prime == 1 );
                const int64_t neg_inv_K = (prime - inv_K) % prime;
                assert( (neg_inv_K * base_r) % prime == (prime-1) );

                host_primes.push_back(prime);
                // host_remainders.push_back(base_r);
                host_neg_inv_Ks.push_back(neg_inv_K);
                num_primes += 1;
            }

            const size_t bytes = sizeof(uint32_t) * num_primes;
            CUDA_CHECK(cudaMallocAsync(&primes, bytes, runner));
            CUDA_CHECK(cudaMemcpyAsync(primes, host_primes.data(), bytes, cudaMemcpyHostToDevice, runner));

            // CUDA_CHECK(cudaMallocAsync(&remainders, bytes, runner));
            // CUDA_CHECK(cudaMemcpyAsync(remainders, host_remainders.data(), bytes, cudaMemcpyHostToDevice, runner));

            CUDA_CHECK(cudaMallocAsync(&neg_inv_Ks, bytes, runner));
            CUDA_CHECK(cudaMemcpyAsync(neg_inv_Ks, host_neg_inv_Ks.data(), bytes, cudaMemcpyHostToDevice, runner));
            // After reading https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
            // I believe memcpyAsync is only async for GPU and CPU is sync with respect to the host.
        }

#ifdef BIT_IS_BIT
        composite_bytes = sizeof(char) * (config.m_inc + 8) / 8 + 1;
#else
        composite_bytes = sizeof(char) * (config.m_inc) + 1;
#endif  // BIT_IS_BIT
        CUDA_CHECK(cudaMallocAsync(&composite, composite_bytes, runner));

        {
            CUDA_CHECK(cudaMallocHost((void**) &host_thread_stats, thread_stats_bytes));
            CUDA_CHECK(cudaMallocAsync(&thread_stats, thread_stats_bytes, runner));
            CUDA_CHECK(cudaMemsetAsync(thread_stats, 0, thread_stats_bytes, runner));
        }

        host_composite_bytes = composite_bytes;
        cudaMallocHost((void**) &host_composite, host_composite_bytes);

        if (config.verbose >= 1) {
            printf("\tGPUSieve(): malloced: primes: 2x %'d  composite: %lu MB + %lu MB\n",
                    4*num_primes, composite_bytes / 1024 / 1024, host_composite_bytes / 1024 / 1024);
        }

        cudaStreamSynchronize(runner);
        auto T1 = high_resolution_clock::now();
        auto gpu_setup_ms = duration_cast<milliseconds>(T1 - T0).count();
        printf("GPU<<<%d,%d>>> setup: %lu ms\n", GRID_SIZE, BLOCK_SIZE, gpu_setup_ms);
    }
}

GPUSieve::~GPUSieve() {
    printf("\t~GPUSieve\n");
    CUDA_CHECK(cudaFreeHost(host_thread_stats));
    CUDA_CHECK(cudaFree(thread_stats));

    //CUDA_CHECK(cudaFree(is_coprime2310));
    CUDA_CHECK(cudaFree(composite));

    CUDA_CHECK(cudaFree(primes));
    // CUDA_CHECK(cudaFree(remainders));
    CUDA_CHECK(cudaFree(neg_inv_Ks));

    CUDA_CHECK(cudaFreeHost(host_composite));
    CUDA_CHECK(cudaStreamDestroy(runner));

    mpz_clear(K);
}

uint8_t* GPUSieve::run(const uint64_t m_start, const uint64_t m_inc, const uint64_t X) {
    { // Run GPU Sieve!
        auto T0 = high_resolution_clock::now();
        CUDA_CHECK(cudaMemsetAsync(composite, 0, composite_bytes, runner));
        method2_medium_primes_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
            this->thread_stats,

            m_start,
            m_inc,
            X,

            this->composite,

            this->num_primes,
            this->primes,
            // this->remainders,
            this->neg_inv_Ks
        );

        cudaStreamSynchronize(runner);

        auto T1 = high_resolution_clock::now();
        auto kernel_ms = duration_cast<milliseconds>(T1 - T0).count();
        //cout << "GPU sieve: " << kernel_ms << " ms" << endl;
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
            auto t0 = s[0];
            auto t1 = s[1];
            auto small_factors = s[2];
            auto verify = s[3];
            if (ti % 173 == 0) {
                printf("\tt%-5lu | t0 offset = %-13ld | t1-t0 = %-12ld | factors: %ld\n",
                        ti, t0 - first_t0, t1 - t0, small_factors);
            }
            assert(verify == ti);
        }
    }

    if (1) { // Parse results back to composite.
        auto T0 = high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpyAsync(host_composite, composite, composite_bytes,
                   cudaMemcpyDeviceToHost, runner));
        cudaStreamSynchronize(runner);

        // Handle 2 manually here
        assert( mpz_odd_p(K) );
        for (uint32_t mi = ((X + m_start) % 2); mi < m_inc; mi += 2) {
#ifdef BIT_IS_BIT
            host_composite[mi >> 3] = 1 << (mi & 7);
#else
            host_composite[mi] = 1;
#endif  // BIT_IS_BIT
        }

#ifdef BIT_IS_BIT
        uint32_t num_composite = 0;
        for (uint32_t mi = 0; mi < host_composite_bytes; mi++) {
            num_composite += __builtin_popcount(host_composite[mi]);
        }
#else
        uint32_t num_composite = std::count(host_composite, host_composite + host_composite_bytes, 1);
#endif  // BIT_IS_BIT

        auto T1 = high_resolution_clock::now();
        auto bitfiddling_ms = duration_cast<milliseconds>(T1 - T0).count();
        //printf("\tGPU copy-back: %lu ms | %u/%lu composite\n", bitfiddling_ms, num_composite, m_inc);
    }

    return host_composite;
}
