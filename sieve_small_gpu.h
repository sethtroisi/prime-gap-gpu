// Copyright 2025 Seth Troisi
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

#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <gmp.h>
#include <primesieve.hpp>
#include <boost/dynamic_bitset.hpp>

#include "gap_common.h"
#include "cached.h"

using std::cout;
using std::endl;
using std::vector;
using boost::dynamic_bitset;
using namespace std::chrono;


// TODO figure out what to set here
#define GRID_SIZE 1024
#define BLOCK_SIZE 64


// BIT_IS_BIT means 8x less GPU memory, faster transfers, slightly slower compute
// A very small percentage of factors get lost.
#define BIT_IS_BIT


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


__device__ uint32_t mod2310(uint32_t n) {
    // magic constant from "Division by Invariant Integers using Multiplication"
    const uint32_t magic_2310 = 1903916239;
    const int32_t shift = 10;
    // mult.hi.u32 = upper 32 bits of u32, u32 multiplication
    uint32_t q;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(q) : "r"(n), "r"(magic_2310));
    q >>= shift;
    //assert( q == (m_temp / 2310) );
    return n - q * 2310;
}


/** Called by host executed on device. */
__global__ void method2_medium_primes_kernal(
    int64_t *thread_stats,

    /** Caches section **/
    char *is_m_coprime2310,
    char *is_coprime2310,
    // Maybe later? is_m_coprime
    int32_t *m_reindex,
    /** End caches section */

    /** config section **/
    const uint64_t M_start,
    const uint32_t M_inc,
    const uint32_t K_mod2310,

    char *composite,
    size_t composite_bytes,

    uint32_t num_primes,
    uint32_t *primes,
    // uint32_t *remainders, // r = K mod p
    int32_t *neg_inv_Ks,  // r^1 mod p

    uint32_t num_coprimes,
    uint32_t *coprime_X_thread
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

    uint32_t m_start_mod2310 = M_start % 2310;

    for (uint32_t pi = blockIdx.x; pi < num_primes; pi += GRID_SIZE) {
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

        for (size_t cxti = threadIdx.x; cxti < num_coprimes; cxti += BLOCK_SIZE) {
        //for (size_t cxti = blockIdx.x; cxti < num_coprimes; cxti += GRID_SIZE) {
            int64_t X = coprime_X_thread[cxti];
            // Safe from overflow as (SL * prime + prime) < int64
            int64_t mi_0 = (X * neg_inv_K + mi_0_shift) % prime;
            mi_0 += (((X ^ mi_0) & 1) == M_parity_check) ? prime : 0;

            uint32_t mi = mi_0;
            for (; mi < M_inc; mi += shift) {
                // TODO if (...) continue code only makes GPU 10% faster
                // Maybe replace with is_m_coprime will be a better check, given fast memory.


                //uint64_t m = M_start + mi;
                //uint32_t m_mod2310 = m % 2310;
                uint32_t m_mod2310 = mod2310(m_start_mod2310 + mi);

                // Filters ~80% or more of m where (m, D) != 1
                if (!is_m_coprime2310[m_mod2310])
                    continue;

                // After initial value this increases by (shift * K_mod2310) % 2310
                //uint32_t n_mod2310 = ((K_mod2310 * m_mod2310) + X) % 2310;
                uint32_t n_mod2310 = mod2310((K_mod2310 * m_mod2310) + X);

                if (!is_coprime2310[n_mod2310])
                    continue;

                // TODO re-add is_m_coprime test.

                int32_t mii = m_reindex[mi];
                if (mii < 0)
                    continue;

                size_t index = (size_t) mii * num_coprimes + cxti;
#ifdef BIT_IS_BIT
                composite[index >> 3] |= 1 << (index&7);
#else
                composite[index] = true;
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

class GPUSieve {
    public:
        cudaStream_t runner;

        // GPU stats
        const size_t   stats_per_thread = 4;
        const size_t   thread_stats_bytes = sizeof(int64_t) * stats_per_thread * GRID_SIZE * BLOCK_SIZE;
        int64_t  *host_thread_stats;
        int64_t  *thread_stats;

        // Cache stuff
        uint32_t num_coprimes;
        uint32_t *coprime_X;

        char     *is_coprime2310;
        char     *is_m_coprime2310;
        int32_t  *m_reindex;

        // Cached prime stuff
        uint32_t num_primes;
        uint32_t *primes;
        // Remainders isn't actually used!
        // uint32_t *remainders; // r = K mod p
        int32_t *neg_inv_Ks;  // r^1 mod p

        // Maybe later? is_m_coprime
        char *composite;
        size_t composite_bytes;

        // Host side

        const size_t composite_segment_size = 30'000'000;
        size_t host_composite_bytes;
        char* host_composite;

        uint64_t M_start;
        uint32_t K_mod2310;

        ~GPUSieve() {
            printf("\t~GPUSieve\n");
            CUDA_CHECK(cudaFreeHost(host_thread_stats));
            CUDA_CHECK(cudaFree(thread_stats));

            CUDA_CHECK(cudaFree(coprime_X));
            CUDA_CHECK(cudaFree(is_coprime2310));
            CUDA_CHECK(cudaFree(is_m_coprime2310));
            CUDA_CHECK(cudaFree(m_reindex));
            CUDA_CHECK(cudaFree(composite));

            CUDA_CHECK(cudaFree(primes));
            // CUDA_CHECK(cudaFree(remainders));
            CUDA_CHECK(cudaFree(neg_inv_Ks));

            CUDA_CHECK(cudaFreeHost(host_composite));
            CUDA_CHECK(cudaStreamDestroy(runner));
        }

        GPUSieve(
                    const Config &config,
                    const mpz_t &K,
                    const Cached &caches,
                    const uint64_t prime_end
        ) {
            auto T0 = high_resolution_clock::now();

            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaStreamCreate(&runner));

            // TODO: only use this if batch takes > 100ms
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
            //CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));


            { // Compute prime stuff and copy over
                vector<uint32_t> host_primes;
                // vector<uint32_t> host_remainders;
                vector<int32_t> host_neg_inv_Ks;

                primesieve::iterator iter(2); // Ignore 2 which is weird
                uint32_t prime = iter.next_prime();
                num_primes = 0;
                for (; prime <= prime_end; prime = iter.next_prime()) {
                    if (prime <= config.p) {
                        continue;
                        if (config.d % prime != 0) {
                            // K % p == 0  ->  Handled by coprime_X
                            continue;
                        }
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
                // I believe memcpyAsync is only async for GPU and CPU is sync with respoct to the host.
            }

            num_coprimes = caches.coprime_X.size();
            const size_t X_bytes = sizeof(uint32_t) * num_coprimes;
            CUDA_CHECK(cudaMallocAsync(&coprime_X, X_bytes, runner));
            CUDA_CHECK(cudaMemcpyAsync(coprime_X, (void*)caches.coprime_X.data(), X_bytes, cudaMemcpyHostToDevice, runner));

            CUDA_CHECK(cudaMallocAsync(&is_coprime2310, 2310, runner));
            CUDA_CHECK(cudaMemcpyAsync(is_coprime2310, (void*)caches.is_coprime2310.data(), 2310, cudaMemcpyHostToDevice, runner));

            {
                // From bitset to char
                char is_m_coprime2310_tmp[2310];
                for (size_t i = 0; i < 2310; i++) is_m_coprime2310_tmp[i] = caches.is_m_coprime2310[i];
                CUDA_CHECK(cudaMallocAsync(&is_m_coprime2310, 2310, runner));
                CUDA_CHECK(cudaMemcpyAsync(is_m_coprime2310, is_m_coprime2310_tmp, 2310, cudaMemcpyHostToDevice, runner));
            }

            const size_t m_reindex_bytes = sizeof(int32_t) * caches.m_reindex.size();

            CUDA_CHECK(cudaMallocAsync(&m_reindex, m_reindex_bytes, runner));
            CUDA_CHECK(cudaMemcpyAsync(m_reindex, caches.m_reindex.data(), m_reindex_bytes, cudaMemcpyHostToDevice, runner));

#ifdef BIT_IS_BIT
            composite_bytes = sizeof(char) * num_coprimes * caches.valid_ms / 8 + 1;
#else
            composite_bytes = sizeof(char) * num_coprimes * caches.valid_ms;
#endif  // BIT_IS_BIT
            CUDA_CHECK(cudaMallocAsync(&composite, composite_bytes, runner));
            CUDA_CHECK(cudaMemsetAsync(composite, 0, composite_bytes, runner));

            {
                CUDA_CHECK(cudaMallocHost((void**) &host_thread_stats, thread_stats_bytes));
                CUDA_CHECK(cudaMallocAsync(&thread_stats, thread_stats_bytes, runner));
                CUDA_CHECK(cudaMemsetAsync(thread_stats, 0, thread_stats_bytes, runner));
            }

            host_composite_bytes = sizeof(char) * composite_segment_size * num_coprimes;
#ifdef BIT_IS_BIT
            assert( composite_segment_size % 8 == 0 ); // needs to be true for segment_bytes to work
            host_composite_bytes /= 8;
#endif  // BIT_IS_BIT
            cudaMallocHost((void**) &host_composite, host_composite_bytes);

            if (config.verbose >= 1) {
                printf("\tGPUSieve(): malloced: primes: 3x %'d  composite: %lu MB + %lu MB\n",
                        4*num_primes, composite_bytes / 1024 / 1024, host_composite_bytes / 1024 / 1024);
            }

            M_start   = config.mstart;
            K_mod2310 = caches.K_mod2310;

            cudaStreamSynchronize(runner);
            auto T1 = high_resolution_clock::now();
            auto gpu_setup_ms = duration_cast<milliseconds>(T1 - T0).count();
            printf("GPU<<<%d,%d>>> setup: %lu ms\n", GRID_SIZE, BLOCK_SIZE, gpu_setup_ms);
        }

        void run_sieve(
            const Config &config,
            uint64_t M_start, uint32_t M_inc,
            const Cached &caches, vector<bool> &output_composite
        ) {
            assert( M_start == this->M_start );

            { // Run GPU Sieve!
                auto T0 = high_resolution_clock::now();
                method2_medium_primes_kernal<<<GRID_SIZE, BLOCK_SIZE, 0, runner>>>(
                    this->thread_stats,

                    this->is_m_coprime2310,
                    this->is_coprime2310,
                    // Maybe later? is_m_coprime
                    this->m_reindex,

                    M_start,
                    M_inc,
                    this->K_mod2310,

                    this->composite,
                    this->composite_bytes,

                    this->num_primes,
                    this->primes,
                    // this->remainders,
                    this->neg_inv_Ks,

                    this->num_coprimes,
                    this->coprime_X // TODO pass a portion of this or something IDK

                );

                cudaStreamSynchronize(runner);

                auto T1 = high_resolution_clock::now();
                auto kernel_ms = duration_cast<milliseconds>(T1 - T0).count();
                cout << "GPU sieve: " << kernel_ms << " ms" << endl;
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

                size_t found_factors = 0;
                //size_t initial_factors =
                //    std::count(output_composite.begin(), output_composite.end(), 1);

                const size_t valid_m = caches.valid_ms;
                char* composite_start = composite;
                for (size_t start_mii = 0; start_mii < valid_m; start_mii += composite_segment_size) {
                    size_t last_mii = std::min(start_mii + composite_segment_size, valid_m);
                    //if (config.verbose >= 3) {
                    //    printf("\tCopying over mi [%lu, %lu)\n", start_mii, last_mii);
                    //}

#ifdef BIT_IS_BIT
                    size_t chunk_bytes = (last_mii - start_mii) * num_coprimes * sizeof(char) / 8;
#else
                    size_t chunk_bytes = (last_mii - start_mii) * num_coprimes * sizeof(char);
#endif  // BIT_IS_BIT
                    assert( 0 < chunk_bytes && chunk_bytes <= host_composite_bytes );
                    assert( chunk_bytes == host_composite_bytes || last_mii == valid_m );
                    CUDA_CHECK(cudaMemcpyAsync(host_composite, composite_start, chunk_bytes, cudaMemcpyDeviceToHost, runner));
                    // TODO is this needed?
                    cudaStreamSynchronize(runner);
                    composite_start += chunk_bytes;

                    // TODO what would it take to use composite directly
                    // so this could just be a memset?
                    // TODO could do something smart like build up chunks of dynamic bitset and commit them.
                    /**
                    size_t had_factor = 0;
                    #pragma omp parallel for schedule(static, 8) num_threads(config.threads) reduction(+:had_factor)
                    for(size_t mii = start_mii; mii < last_mii; mii++) {
                        size_t offset = (mii - start_mii) * num_coprimes;
                        uint64_t m = M_start + caches.valid_mi[mii];
                        const uint32_t m_mod_wheel = m % caches.x_reindex_wheel_size;
                        // TODO test with and without &
                        const auto x_reindex_m = caches.x_reindex_wheel.data() + ((m_mod_wheel) * SL_PLUS1);
                        size_t m_offset = mii * caches.composite_line_size;

                        for (size_t xi = 0; xi < num_coprimes; xi++, offset++) {
                            // if [mii][xi] is composite
#ifdef BIT_IS_BIT
                            if (host_composite[offset >> 3] & (1 << (offset & 7))) {
#else
                            if (host_composite[offset]) {
#endif  // BIT_IS_BIT
                                had_factor += 1;
                                auto X = caches.coprime_X[xi];
                                auto xii = x_reindex_m[X];
                                if (xii > 0) {
                                    output_composite[m_offset + xii] = 1;
                                }
                            }
                        }
                    }
                    if (config.verbose >= 3) {
                        printf("\tmi [%lu, %lu) -> %lu factors\n", start_mii, last_mii, had_factor);
                    }
                    found_factors += had_factor;
                    */
                }

                //size_t after_factors =
                //    std::count(output_composite.begin(), output_composite.end(), 1);
                //printf("\t%lu + %lu -> %lu = %lu new\n",
                //    initial_factors, found_factors, after_factors, after_factors - initial_factors);

                auto T1 = high_resolution_clock::now();
                auto bitfiddling_ms = duration_cast<milliseconds>(T1 - T0).count();
                printf("\tGPU copy-back: %lu ms | %lu factors\n", bitfiddling_ms, found_factors);
            }
        }
};
