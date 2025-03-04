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
#include "sieve_small_gpu.h"


using std::vector;

class SieveOutput {
    public:
        // Prevent copying which would use lots of memory...
        SieveOutput(const SieveOutput&) = delete;
        void operator=(const SieveOutput&) = delete;

        SieveOutput(const struct Config& config);

        /* Was used for debug.
        ~SieveOutput() {
            printf("~SieveOutput\n");
        }
        // */

        // Used to run the sieve
        GPUSieve *gpu_sieve;

        uint64_t m_start;

        vector<uint16_t> coprime_X;

        // mi for m being considered, set to -1 to remove a values
        vector<uint32_t> m_inc;

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
};
