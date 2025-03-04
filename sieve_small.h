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

#include <bitset>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "gap_common.h"

// TODO figure out what to set here
#define GRID_SIZE 1024
#define BLOCK_SIZE 64

using std::vector;

class GPUSieve {
    public:
        // Prevent copying which would use lots of memory...
        GPUSieve(const GPUSieve&) = delete;
        void operator=(const GPUSieve&) = delete;

        GPUSieve(const struct Config& config);
        ~GPUSieve();

        vector<uint16_t> coprime_X;

        uint64_t M_start;
        // mi for m being considered, set to -1 to remove a values
        vector<int32_t> m_inc;

        /**
         * 32 bits of "unknowns"
         * 0th bit is coprime_X[unknown_X0], 1st bit is coprime_X[unknown_X0+1] ...
         */
        uint16_t unknown_X0;
        vector<uint32_t> unknowns;

        // The largest X evaluated by this sieve.
        int32_t  max_X;

        void update(const struct Config& new_config);
        void run(const struct Config& config);

    private:
        cudaStream_t runner;

        // TODO consider moving to GPUCached
        /**** GPU POINTERS ****/
        // GPU stats
        const size_t   stats_per_thread = 4;
        const size_t   thread_stats_bytes = sizeof(int64_t) * stats_per_thread * GRID_SIZE * BLOCK_SIZE;
        int64_t  *host_thread_stats;
        int64_t  *thread_stats;

        // Cache stuff
        uint32_t num_coprimes;
        uint32_t *test_X;

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
        /**** END GPU POINTERS ****/

        // Host side

        const size_t composite_segment_size = 30'000'000;
        size_t host_composite_bytes;
        char* host_composite;

        mpz_t K;
        uint32_t K_mod2310;

        // TODO implement this
        uint64_t M_start_check;

};
