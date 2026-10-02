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

#include "gap_search_common.h"
#include "gpu_testing.h"


// TODO consider if I can avoid uint64_t for some of these (jump can be)
struct LinearRange {
    // An interval is the current [offset, offset + inc]
    // A "Range" is a fraction of the whole composite [start, end)
    uint64_t range_start = 0;
    uint64_t range_end = 0;

    // These are all half indexes (e.g. divide by two)
    uint64_t start = 0;
    // Can probably avoid storing this with a little work.
    uint64_t jump = 0;
    // Test number is: K + TestData.offset + current
    uint64_t current = 0;

    uint8_t in_gpu_batch = 0;
    enum State : uint8_t { NEW, PHASE1, PHASE2, PRIME, DONE };
    State state;
};

class LinearTestData {
    public:
        LinearTestData(const struct Config config);

        /**
         * WAITING -> ACTIVE -> DONE
         *    ^                  |
         *    |------------------v
         */
        enum State { WAITING, ACTIVE, DONE };
        std::atomic<State> state = WAITING;

        // From Config
        int verbose;

        // For current range
        uint64_t offset = 0;
        uint64_t length = 0;

        mpz_t test_k;

        vector<LinearRange> ranges;

        vector<uint32_t> composites;

        std::atomic<uint32_t> running_batches = 0;
        std::atomic<uint32_t> active_batches = 0;

        std::atomic<uint32_t> active_ranges = 0;

        // Stats
        StatsCounters stats;
        GpuStatsCounters gpu_stats;

        // Methods
        void reset();
        void setup_ranges();

        /** Should hold lock during */
        void maybe_print_stats() {
            uint64_t c = stats.batches;
            bool is_power_print = false;
            for (uint64_t p = 1; p <= c; p *= 10) {
                is_power_print |= (c == p) || (c == 2*p) || (c == 5*p);
            }
            if (is_power_print) {
                print_stats();
            }
        }

        /** Should hold lock during */
        void print_stats();

        void lock();
        void unlock();
        void wait_for_state_and_lock(State desired);

    private:
        // For signaling, must be owned to change state.
        /**
         * :wait(0) -> unlock
         * -> set to 1 to lock with a check?
         */
        std::atomic<int> flag;
};


void run_gpu_thread(int runner_num, int verbose,
                    LinearTestData &test_data, GPUBatch& batch,
                    const mpz_t &K_in);
