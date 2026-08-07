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

const size_t COPRIME_PER = 32;

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

    uint32_t num_primes,
    uint32_t *primes,
    // uint32_t *remainders, // r = K mod p
    int32_t *neg_inv_Ks,  // r^1 mod p

    uint16_t *coprime_X,
    uint32_t coprime_X0
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

    assert( BLOCK_SIZE == 64 );
    assert( COPRIME_PER == 32 );
    uint32_t pi_0 = (blockIdx.x << 1) + (threadIdx.x & 1);

    // 32 coprime_X0 starting with coprime_X0, note that indexing is always 0..31
    // cxti = coprime X, thread index
    size_t cxti = (threadIdx.x >> 1);
    int64_t X = coprime_X[coprime_X0 + cxti];

#ifdef BIT_IS_BIT
    char index_bit = 1 << (cxti & 7);
#endif  // BIT_IS_BIT

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
                //uint32_t n_mod2310 = mod2310((K_mod2310 * m_mod2310) + X);

                // TODO This isn't safe if we don't post process divisors of d or something.
                //if (!is_coprime2310[n_mod2310])
                //    continue;

                // TODO re-add is_m_coprime test.

                int32_t mii = m_reindex[mi];
                if (mii < 0)
                    continue;

#ifdef BIT_IS_BIT
                // TODO 1<<(index&7) this could be extracted to the top to save a few operations
                composite[(mii << 2) + (cxti >> 3)] |= index_bit;
#else
                size_t index = (size_t) (mii << 5) + cxti;
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


/*void method2_print(
        uint64_t prime,
        size_t valid_ms,
        vector<bool> &composite,
        method2_stats &stats,
        const struct Config& config) {

    auto   s_stop_t = high_resolution_clock::now();
    // total time, interval time
    double     secs = duration<double>(s_stop_t - stats.start_t).count();
    uint32_t SIEVE_LENGTH = config.sieve_length;

    printf("%'-10ld\t(seconds: %-.1f | per m: %.3g)",
        prime, secs, secs / valid_ms);

    // See THEORY.md
    double prob_prime_after_sieve = stats.prob_prime * log(prime) * exp(GAMMA);
    double delta_sieve_prob = (1/stats.current_prob_prime - 1/prob_prime_after_sieve);
    double skipped_prp = valid_ms * delta_sieve_prob;

    uint64_t t_total_unknowns = std::count(composite.begin(), composite.end(), 0);
    uint64_t new_composites = stats.total_unknowns - t_total_unknowns;

    // count_coprime_sieve * valid_ms also makes sense but leads to smaller numbers
    printf("\tunknowns %'9ld/%-5ld\t"
           "(avg/m: %.2f) (composite: %.2f%% +%.3f%% +%'ld)\n",
        t_total_unknowns, valid_ms,
        1.0 * t_total_unknowns / valid_ms,
        100.0 - 100.0 * t_total_unknowns / (SIEVE_LENGTH * valid_ms),
        100.0 * new_composites / (SIEVE_LENGTH * valid_ms),
        new_composites);

    // NOTE: There used to be some validity check on t_total_unknowns ti required
    // count_coprime_p which is hard to estimate and was directly computed when
    // prime pasted P, this is non-trivial in the GPU only world.

    stats.total_unknowns = t_total_unknowns;
    stats.current_prob_prime = prob_prime_after_sieve;
}
*/

/**
 * TODO better name: RangeStats, KStats, Helpers, Indexes?
 * Helper arrays
 */
class Cached {
    public:
        // mi such that gcd(m_start + mi, D) = 1
        // TODO: Can I change this to m_inc as a uint8_t?
        vector<uint32_t> valid_mi;
        // valid_mi.size()
        size_t valid_ms;

        /**
         * m_reindex[mi] = mii (index into composite) if coprime
         * -1 if gcd(ms + i, D) > 1
         *
         * This is potentially very large use is_m_coprime and is_m_coprime2310
         * to pre check it's a coprime mi before doing the L3/RAM lookup.
         */
        vector<int32_t> m_reindex;
        // if gcd(ms + mi, D) = 1
        // TODO try with dynamic_bitset
        //vector<bool> is_m_coprime;
        /**
         * is_m_coprime2310[i] = (i, D') == 1
         * D' = gcd(2310, D)
         * first 2310 values.
         * vector<bool> seems faster than char [2310]
         */
        std::bitset<2310> is_m_coprime2310;


        // X which are coprime to K
        vector<uint16_t> coprime_X;
        // reindex composite[m][X] for composite[m_reindex[m]][x_reindex[X]]
        // Special 0'th entry stands for all not coprime
        //vector<uint32_t> x_reindex;

        uint64_t composite_line_size;

        int32_t K_mod2310;

        // is_comprime2310[i] = (i % 2) && (i % 3) && (i % 5) && (i % 7) && (i % 11)
        vector<char> is_coprime2310;

        Cached(const struct Config& config, const mpz_t &K);
};

Cached::Cached(const struct Config& config, const mpz_t &K) {
        const uint32_t P = config.p;
        const uint32_t D = config.d;

        const uint32_t SL = config.sieve_length;

        const vector<uint32_t> P_primes = get_sieve_primes(P);
        assert( P_primes.back() == P);

        // Allocate temp vectors
        m_reindex.resize(config.minc, -1);
        {
            auto temp = is_coprime_and_valid_m(config);
            //is_m_coprime = temp.first;
            valid_mi = temp.second;

            for (uint32_t mii = 0; mii < valid_mi.size(); mii++) {
                uint32_t mi = valid_mi[mii];
                m_reindex[mi] = mii;
            }
        }
        valid_ms = valid_mi.size();

        // Includes 0
        //x_reindex.resize(SL+1, 0);

        // reindex composite[m][i] using coprime_X

        vector<char> is_offset_coprime(SL+1, 1);
        for (uint32_t prime : P_primes) {
            if (D % prime != 0) {
                for (size_t x = 0; x <= SL; x += prime) {
                    is_offset_coprime[x] = 0;
                }
            }
        }
        assert( D % 2 == 0 );
        // 2 | D <=> K is odd <=> all m % 2 == 1
        // m % 2 == 1 -> any odd x will be divisible by two
        // These can't have unknowns so skip by marking as not-coprime.
        // TODO backport this logic assuming it's valid.
        if (1) {
            size_t skipped = 0;
            for (size_t x = 1; x <= SL; x += 2) {
                if (is_offset_coprime[x]) {
                    is_offset_coprime[x] = 0;
                    skipped += 1;
                }
            }
            if (config.verbose >= 2) {
                printf("\tSkipped %lu coprime X which are always divisible by 2\n", skipped);
            }
        }

        // Center should be marked composite by every prime.
        assert(is_offset_coprime[0] == 0);
        {
            size_t coprime_count = 0;
            for (size_t X = 0; X <= SL; X++) {
                if (is_offset_coprime[X] > 0) {
                    coprime_X.push_back(X);
                    coprime_count += 1;
                    //x_reindex[X] = coprime_count;
                    //printf("\tcoprime(%lu) = %lu\n", coprime_count, X);
                }
            }
            assert(coprime_count == coprime_X.size());
        }

        composite_line_size = 8 * ((coprime_X.size() + 7) / 8);
        if (config.verbose >= 2) {
            cout << "Need at least " << coprime_X.size() << " per m, rounding up to "
                 << composite_line_size << endl << endl;;
        }

        K_mod2310 = mpz_fdiv_ui(K, 2310);

        is_coprime2310.resize(2*3*5*7*11, 1);
        for (int p : {2, 3, 5, 7, 11})
            for (size_t i = 0; i < is_coprime2310.size(); i += p)
                is_coprime2310[i] = 0;

        //is_m_coprime2310.resize(2310, 1);
        is_m_coprime2310.set();

        for (int p : {2, 3, 5, 7, 11})
            if (config.d % p == 0)
                for (int i = 0; i < 2310; i += p)
                    is_m_coprime2310[i] = 0;

        assert(count(is_coprime2310.begin(), is_coprime2310.end(), 1) == 480);
};


GPUSieve::GPUSieve(const struct Config& config) {
    // ----- Sieve stats & Merit Stuff
    const double K_log = prob_prime_and_stats(config, K);
    const double N_log = K_log + log(config.mstart);
    const double prob_prime = 1 / N_log - 1 / (N_log * N_log);

    // ----- Allocate memory

    // Various pre-calculated arrays of is_coprime arrays
    Cached caches(config, K);
    const size_t valid_ms = caches.valid_ms;

    const size_t count_coprime_sieve = caches.coprime_X.size();
    if (config.verbose >= 1) {
        printf("\tsieve_length:  %'d\n", config.sieve_length);
        printf("\tmax_prime:     %'ld\n", config.max_prime);
        printf("\tcoprime m:     %'lu / %'lu\n", valid_ms, config.minc);
        printf("\tcoprime i:     %lu / %u\n", count_coprime_sieve, config.sieve_length);
    }

    { // Output setup
        // TODO any other variables?
        M_start       = config.mstart;
        M_start_check = config.mstart;
        K_mod2310 = caches.K_mod2310;
        // Only one copy of these two
        {
            m_inc.swap(caches.valid_mi);
            host_reindex.swap(caches.m_reindex);
        }
        coprime_X = caches.coprime_X;
        unknown_X0 = 0;
        unknowns.resize(valid_ms, 0);
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
                if (prime <= config.p) {
                    if (config.d % prime != 0) {
                        // K % p == 0  ->  Handled by coprime_X
                        continue;
                    }
                    // TODO do something for d primes
                    // maybe handle afterwards, manually, IDK
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

        num_coprimes = caches.coprime_X.size();
        const size_t X_bytes = sizeof(uint16_t) * num_coprimes;
        CUDA_CHECK(cudaMallocAsync(&test_X, X_bytes, runner));
        CUDA_CHECK(cudaMemcpyAsync(test_X, (void*)caches.coprime_X.data(), X_bytes, cudaMemcpyHostToDevice, runner));

        CUDA_CHECK(cudaMallocAsync(&is_coprime2310, 2310, runner));
        CUDA_CHECK(cudaMemcpyAsync(is_coprime2310, (void*)caches.is_coprime2310.data(), 2310, cudaMemcpyHostToDevice, runner));

        {
            // From bitset to char
            char is_m_coprime2310_tmp[2310];
            for (size_t i = 0; i < 2310; i++) is_m_coprime2310_tmp[i] = caches.is_m_coprime2310[i];
            CUDA_CHECK(cudaMallocAsync(&is_m_coprime2310, 2310, runner));
            CUDA_CHECK(cudaMemcpyAsync(is_m_coprime2310, is_m_coprime2310_tmp, 2310, cudaMemcpyHostToDevice, runner));
        }

#ifdef BIT_IS_BIT
        composite_bytes = sizeof(char) * COPRIME_PER * caches.valid_ms / 8 + 1;
#else
        composite_bytes = sizeof(char) * COPRIME_PER * caches.valid_ms;
#endif  // BIT_IS_BIT
        CUDA_CHECK(cudaMallocAsync(&composite, composite_bytes, runner));

        {
            CUDA_CHECK(cudaMallocHost((void**) &host_thread_stats, thread_stats_bytes));
            CUDA_CHECK(cudaMallocAsync(&thread_stats, thread_stats_bytes, runner));
            CUDA_CHECK(cudaMemsetAsync(thread_stats, 0, thread_stats_bytes, runner));
        }

        host_composite_bytes = sizeof(char) * composite_segment_size * COPRIME_PER;
#ifdef BIT_IS_BIT
        assert( composite_segment_size % 8 == 0 ); // needs to be true for segment_bytes to work
        host_composite_bytes /= 8;
#endif  // BIT_IS_BIT
        cudaMallocHost((void**) &host_composite, host_composite_bytes);

        if (config.verbose >= 1) {
            printf("\tGPUSieve(): malloced: primes: 3x %'d  composite: %lu MB + %lu MB\n",
                    4*num_primes, composite_bytes / 1024 / 1024, host_composite_bytes / 1024 / 1024);
        }

        assert( host_reindex.size() == config.minc );
        const size_t m_reindex_bytes = sizeof(int32_t) * host_reindex.size();
        CUDA_CHECK(cudaMallocAsync(&m_reindex, m_reindex_bytes, runner));
        CUDA_CHECK(cudaMemcpyAsync(m_reindex, host_reindex.data(), m_reindex_bytes, cudaMemcpyHostToDevice, runner));


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

    CUDA_CHECK(cudaFree(test_X));
    CUDA_CHECK(cudaFree(is_coprime2310));
    CUDA_CHECK(cudaFree(is_m_coprime2310));
    CUDA_CHECK(cudaFree(m_reindex));
    CUDA_CHECK(cudaFree(composite));

    CUDA_CHECK(cudaFree(primes));
    // CUDA_CHECK(cudaFree(remainders));
    CUDA_CHECK(cudaFree(neg_inv_Ks));

    CUDA_CHECK(cudaFreeHost(host_composite));
    CUDA_CHECK(cudaStreamDestroy(runner));

    mpz_clear(K);
}

void GPUSieve::update() {
    uint32_t old_min_X = coprime_X[unknown_X0];

    // TODO how to make sure gap_test_gpu doesn't change this
    unknown_X0 += COPRIME_PER;
    if (unknown_X0 + COPRIME_PER > this->num_coprimes) {
        printf("Partial num_coprimes is not handled. use larger sieve_length?\n");
        printf("But also you should never be here. TODO clean up this comment\n");
        assert(false);
    }

    // 1. Check for any m_inc that are no longer needed
    // 2. Update host_reindex and later is_m_coprime
    // 3. Update GPU mirrors.

    // 1.
    size_t old_ms = m_inc.size();
    uint32_t sentinel = std::numeric_limits<uint32_t>::max();
    m_inc.erase(std::remove(m_inc.begin(), m_inc.end(), sentinel), m_inc.end());
    size_t new_ms = m_inc.size();
    printf("GPUSieve::update() [%d to %d], %lu -> %lu | first: %lu\n",
            old_min_X, coprime_X[unknown_X0-1], old_ms, new_ms, M_start + m_inc[0]);

    // 2.
    {
        // TODO maybe later this can be zerod at same time as m_inc is erased
        // for now have to do full write of -1 everywhere
        std::fill(host_reindex.begin(), host_reindex.end(), -1);

        for (uint32_t mii = 0; mii < m_inc.size(); mii++) {
            uint32_t mi = m_inc[mii];
            host_reindex[mi] = mii;
        }
    }

    // 3.
    const size_t m_reindex_bytes = sizeof(int32_t) * host_reindex.size();
    CUDA_CHECK(cudaMemcpyAsync(m_reindex, host_reindex.data(), m_reindex_bytes, cudaMemcpyHostToDevice, runner));
}

void GPUSieve::run(const struct Config& config) {
    // Used for various stats
    // TODO print something.
    //method2_stats stats(new_config, valid_ms, prob_prime);
    //method2_print(LAST_PRIME, caches.valid_ms, composite, stats, config);
    //save_unknowns(config, K, caches, composite);

    uint64_t M_start = config.mstart;
    // TODO M_start_check
    uint32_t M_inc = config.minc;

    assert( M_start == this->M_start );

    assert( unknown_X0 + COPRIME_PER <= this->num_coprimes );
    { // Run GPU Sieve!
        auto T0 = high_resolution_clock::now();
        CUDA_CHECK(cudaMemsetAsync(composite, 0, composite_bytes, runner));
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

            this->num_primes,
            this->primes,
            // this->remainders,
            this->neg_inv_Ks,

            this->test_X,

            this->unknown_X0
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

        const size_t valid_m = m_inc.size();

        char* composite_start = composite;
        for (size_t start_mii = 0; start_mii < valid_m; start_mii += composite_segment_size) {
            size_t last_mii = std::min(start_mii + composite_segment_size, valid_m);

#ifdef BIT_IS_BIT
            size_t chunk_bytes = (last_mii - start_mii) * COPRIME_PER * sizeof(char) / 8;
            // TODO can this be simplified to a straight memory copy?
#else
            size_t chunk_bytes = (last_mii - start_mii) * COPRIME_PER * sizeof(char);
#endif  // BIT_IS_BIT
            if (config.verbose >= 1) {
                printf("\tCopying over mi [%lu, %lu) | %lu bytes\n", start_mii, last_mii, chunk_bytes);
            }
            assert( 0 < chunk_bytes && chunk_bytes <= host_composite_bytes );
            assert( chunk_bytes == host_composite_bytes || last_mii == valid_m );
            CUDA_CHECK(cudaMemcpyAsync(host_composite, composite_start, chunk_bytes, cudaMemcpyDeviceToHost, runner));
            composite_start += chunk_bytes;

            size_t had_factor = 0;
            #pragma omp parallel for schedule(static, 32) num_threads(config.threads) reduction(+:had_factor)
            for(size_t mii = start_mii; mii < last_mii; mii++) {
                size_t offset = (mii - start_mii) * COPRIME_PER;
                uint64_t m = M_start + m_inc[mii];

                uint32_t unknown = 0;
#ifdef BIT_IS_BIT
                // JOIN 4 chars
                size_t t = offset >> 3;
                unknown = ( host_composite[t] |
                           (host_composite[t+1] << 8) |
                           (host_composite[t+2] << 16) |
                           (host_composite[t+3] << 24) );
#else
                // TODO mark this as #pragma GCC unroll 8
                for (size_t xi = 0; xi < COPRIME_PER; xi++) {
                    unknown |= (host_composite[offset + xi] > 0) << xi;
                }
#endif  // BIT_IS_BIT
                // Invert all bits because we want unknowns, not composites.
                unknowns[mii] = unknown ^ 0xFFFFFFFF;
                had_factor += __builtin_popcountl(unknown);
            }
            if (config.verbose >= 3) {
                printf("\tmi [%lu, %lu) -> %lu factors\n", start_mii, last_mii, had_factor);
            }
            found_factors += had_factor;
        }

        auto T1 = high_resolution_clock::now();
        auto bitfiddling_ms = duration_cast<milliseconds>(T1 - T0).count();
        printf("\tGPU copy-back: %lu ms | %lu factors\n", bitfiddling_ms, found_factors);
    }
}
