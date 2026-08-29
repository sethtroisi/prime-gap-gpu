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

// Should be a multiple of SM (66 on 4070)
#define GRID_SIZE (2*66)
// number of threads, multiple of 32
#define BLOCK_SIZE 64

class GPUSieve {
    public:
        // Prevent copying which would use lots of memory...
        GPUSieve(const GPUSieve&) = delete;
        void operator=(const GPUSieve&) = delete;

        GPUSieve(const struct Config& config);
        ~GPUSieve();

        uint8_t* run(const uint64_t m_start, const uint64_t m_inc, const uint64_t X, const uint32_t max_p_i);

    private:
        cudaStream_t runner;

        uint32_t verbose = 0;
        uint64_t number_sieves = 0;
        double   total_sieve_time = 0;

        uint32_t num_primes;
        uint32_t num_small_primes;

        mpz_t K;
        uint32_t D;
        uint32_t K_mod_d;
        size_t d_wheel_bytes;

        const size_t   stats_per_thread = 4;
        const size_t   thread_stats_bytes = sizeof(int64_t) * stats_per_thread * GRID_SIZE * BLOCK_SIZE;
        size_t host_composite_bytes;

        /******** GPU POINTERS ********/
        /******************************/
        // GPU stats
        int64_t  *thread_stats;

        // Both of these are `num_primes` long
        uint32_t *primes;
        int32_t *neg_inv_Ks;  // r^1 mod p

        size_t composite_bytes;
        uint8_t *composite;
        uint8_t *d_wheel;
        /******************************/
        /******** GPU POINTERS ********/

        // Host side
        uint8_t* host_composite;
        int64_t  *host_thread_stats;
        vector<std::pair<uint32_t, uint32_t>> d_neg_inv_K;
        vector<uint8_t> host_wheel;
};
