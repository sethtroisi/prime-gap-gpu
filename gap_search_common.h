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

#include <atomic>
#include <cstdint>
#include <unistd.h>

#include "gap_common.h"
#include "gap_stats.h"

using std::vector;


// GLOBALS PART 1

// TODO add gap_search_common.cpp with these I guess
/** Shared state between threads */
extern std::atomic<bool> is_running;
extern std::atomic<uint8_t> stop_queue;
/**
 * is_running = false
    stop immediately
 * stop_queue
    * 0: everything normal
    * 1: continue like normal till increment_m
    * 2: stop sieve & gpu_tester
         wait for overflow to finish
 */

// Overflow globals in overflow.h

class TestData {
    public:
        TestData(const struct Config config);

        /**
         * WAITING -> ACTIVE -> DONE
         *    ^                  |
         *    |------------------v
         */
        enum State { WAITING, ACTIVE, DONE };
        std::atomic<State> state = WAITING;

        // From Config
        int verbose;
        uint32_t m_inc;

        // For current range
        uint64_t m_start = 0;
        uint32_t testing_x = 0;

        vector<uint32_t> unknown_m_i;
        // all indexes < test_i have been queued in a GPUBatch
        size_t test_i = 0;

        std::atomic<uint32_t> running_batches = 0;
        std::atomic<uint32_t> active_batches = 0;

        /* BITSET of half of m_i where a prime has been found (at any X). */
        vector<uint32_t> found_prime_m_i;

        // Stats
        StatsCounters stats;
        GpuStatsCounters gpu_stats;

        // Methods
        void full_reset();

        void add_found_prime_m_i(const uint32_t m_i) {
            //assert(m_i < m_inc);
            // all m_i are even (see sieve) so shift down by 1
            uint32_t t = m_i >> 1;
            found_prime_m_i[t >> 5] |= 1 << (t & 31);
        }

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
