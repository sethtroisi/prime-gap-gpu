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

#include <bitset>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "gap_common.h"

// TODO figure out what to set here
#define GRID_SIZE 64
#define BLOCK_SIZE 32 // Number of threads

using std::vector;

class GPUSieve {
    public:
        // Prevent copying which would use lots of memory...
        GPUSieve(const GPUSieve&) = delete;
        void operator=(const GPUSieve&) = delete;

        GPUSieve(const struct Config& config);
        ~GPUSieve();

        uint8_t* run(const uint64_t m_start, const uint64_t m_inc, const uint64_t X);

    private:
        cudaStream_t runner;

        uint32_t verbose = 0;
        uint64_t number_sieves = 0;
        double   total_sieve_time = 0;

        /**** GPU POINTERS ****/
        // GPU stats
        const size_t   stats_per_thread = 4;
        const size_t   thread_stats_bytes = sizeof(int64_t) * stats_per_thread * GRID_SIZE * BLOCK_SIZE;
        int64_t  *host_thread_stats;
        int64_t  *thread_stats;

        // Cache stuff

        //char     *is_coprime2310;

        // Cached prime stuff
        uint32_t num_primes;
        uint32_t num_small_primes;
        uint32_t *primes;
        // Remainders isn't actually used!
        // uint32_t *remainders; // r = K mod p
        int32_t *neg_inv_Ks;  // r^1 mod p

        size_t composite_bytes;
        uint8_t *composite;
        /**** END GPU POINTERS ****/

        // Host side
        size_t host_composite_bytes;
        uint8_t* host_composite;

        mpz_t K;
        //uint32_t K_mod2310;
};
