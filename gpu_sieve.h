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

#pragma once

#include <cstdint>
#include <utility>
#include <cuda_runtime.h>

#include "gap_common.h"

// Should be a multiple of streaming multi proccessors (66 on 4070)
#define GRID_SIZE (4 * MP)
// number of threads, multiple of 32
#define BLOCK_SIZE 256

// Most X sieved by one pass over the large primes
#define MAX_SIEVE_BATCH 8

class GPUSieve {
    public:
        // Prevent copying which would use lots of memory...
        GPUSieve(const GPUSieve&) = delete;
        void operator=(const GPUSieve&) = delete;

        GPUSieve(const struct Config& config);
        ~GPUSieve();

        /**
         * Sorted odd mi (m = m_start + mi) that are active and not composite for X,
         * valid till the next call of run / run_unknowns.
         */
        const uint32_t* run_unknowns(
                const uint64_t m_start, const uint64_t m_inc, const uint64_t X, const uint32_t max_p_i,
                uint32_t *count);

        /**
         * True if run_unknowns() would sieve a new batch, call set_active_bits() first if so.
         * The rest of the batch reuses those (slightly stale) active bits.
         */
        bool wants_active_bits(const uint64_t m_start, const uint64_t X, const uint32_t max_p_i) const {
            return find_cached(m_start, X, max_p_i, true) < 0;
        }
        /** bit i => m_start + 2 * i + 1 still needs testing, bits past m_inc / 2 must be zero. */
        void set_active_bits(const uint64_t m_start, const uint64_t *bits, const size_t words);

        /**
         * Composite bits for X (bit i => m_start + 2 * i + 1) without finalize,
         * valid till the next call of run / run_unknowns.
         */
        uint64_t* run(const uint64_t m_start, const uint64_t m_inc, const uint64_t X, const uint32_t max_p_i);

        uint32_t get_num_primes() const { return num_primes; }

        /** Sync after each kernel for per kernel timing (slightly slower). */
        void set_profile(bool enable) { profile = enable; }

        /**
         * X and the next (batch - 1) coprime X are sieved together (1 disables),
         * later calls for the same m_start and one of those X return the stored result.
         */
        void set_batch_size(uint32_t batch);
        uint32_t get_batch_size() const { return batch_size; }

    private:
        /** Sieve (and finalize if with_unknowns) first_X and the following coprime X. */
        void sieve_batch(
                const uint64_t m_start, const uint64_t first_X, const uint32_t max_p_i, bool with_unknowns);

        /** Index into cache_X, after sieving a new batch if needed. */
        uint32_t cached_or_sieve(
                const uint64_t m_start, const uint64_t m_inc,
                const uint64_t X, const uint32_t max_p_i, bool with_unknowns);

        /** Index into cache_X or -1. */
        int32_t find_cached(
                const uint64_t m_start, const uint64_t X, const uint32_t max_p_i, bool need_unknowns) const;

        /** D wheel for each X of cache_X. */
        void upload_wheels(const uint64_t m_start);

        cudaStream_t runner;

        uint32_t verbose = 0;
        bool profile = false;

        uint64_t number_sieves = 0;
        uint64_t batches_run = 0;
        uint64_t unknown_fallbacks = 0;
        double   d_total    = 0.0;
        double   d_wheel    = 0.0;
        double   d_segment  = 0.0;
        double   d_medium   = 0.0;
        double   d_large    = 0.0;
        double   d_finalize = 0.0;
        double   d_copy     = 0.0;

        uint64_t m_inc;
        mpz_t K;
        uint32_t D;
        uint32_t K_mod_d;
        // (prime, -(K^-1) mod prime) for odd primes dividing D
        vector<std::pair<uint32_t, uint32_t>> d_neg_inv_K;
        // Product of the odd primes dividing D, period of the wheel.
        uint32_t wheel_size;

        uint32_t num_primes;
        uint32_t num_segment_primes; // primes handled by segment_kernel
        uint32_t num_medium_primes;  // primes < m_inc/2

        // Bits (bytes in shared memory) per block of segment_kernel
        uint32_t segment_size;
        uint32_t num_segments;

        /******** Batch / cache ********/

        uint32_t batch_size = 1;
        // Used to find the X that will (probably) be sieved next.
        vector<uint32_t> coprime_X;
        // Parameters that produced composite[i] / unknowns[i]
        uint64_t cache_m_start = 0;
        uint32_t cache_max_p_i = 0;
        vector<uint64_t> cache_X;
        bool cache_has_unknowns = false;

        /******** GPU POINTERS ********/

        // Both of these are `num_primes` long
        uint32_t *primes;
        uint32_t *neg_inv_Ks;  // -(K^-1) mod p

        // MAX_SIEVE_BATCH arrays each composite_bytes long, bit i set => m_start + 2 * i + 1 is composite
        size_t composite_bytes;
        uint32_t *composite;

        // MAX_SIEVE_BATCH arrays each wheel_size long, wheel_words[r] = bits [r, r + 32) of the D wheel
        uint32_t *wheel_words;

        // composite_bytes long (zero padded)
        uint32_t *active;
        // Exclusive prefix sum of unknowns per chunk (scratch for unknown_*_kernel)
        uint32_t num_chunks;
        uint32_t *chunk_offsets;
        // MAX_SIEVE_BATCH arrays each unknown_capacity long, and their counts
        uint32_t unknown_capacity;
        uint32_t *unknowns;
        uint32_t *unknown_counts;

        /******** Pinned host memory ********/

        // Bytes of composite / active the host uses (m_inc / 2 bits rounded up to uint64)
        size_t copy_bytes;
        uint64_t *host_composite;
        uint64_t *host_active;
        uint64_t host_active_m_start = 0;
        bool host_active_pending = false;
        // Same layout as unknowns / unknown_counts
        uint32_t *host_unknowns;
        uint32_t *host_unknown_counts;
        // Used if there are more than unknown_capacity unknowns
        vector<uint32_t> host_unknowns_fallback;
};
