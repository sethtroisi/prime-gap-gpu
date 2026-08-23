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
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

#include <gmp.h>

#include "gap_common.h"
#include "gap_stats.h"

using std::vector;
using namespace std::chrono;


// GLOBALS PART 1

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

extern std::mutex overflow_mtx;
extern std::condition_variable overflow_cv;

// Overflow glabls in overflow.h

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
            // s_gap_out_of_sieve_prev is actually count of intervals
            uint64_t c = stats.s_gap_out_of_sieve_prev;
            bool is_power_print = false;
            for (uint64_t p = 1000; p <= c; p *= 10) {
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

class SieveData {
    public:
        SieveData(const struct Config config);

        /**
         * NEW -> ACTIVE -> FINAL -> DONE
         * FIRST_SIEVE => Running the first sieve
         * FINAL => Don't sieve any more, just finish outstanding prime tests.
         *      would be kinda nice to start on next sieves but IDK how to avoid that delay.
         */
        enum State { NEW, FIRST_SIEVE, ACTIVE, FINAL, DONE };
        State state = NEW;

        struct Config config;

        /* Number of valid m_i for [m_start, m_start + m_inc). */
        size_t num_valid = 0;

        vector<uint32_t> coprime_X;

        size_t testing_x_i = 0;
        size_t current_testing_x = 0;
        /**
         * m_i values without a prime
         * this list corresponds to numbers BEFORE current_testing_x
         */
        vector<uint32_t> active_m_i;

        size_t sieve_x_i = 0;
        size_t current_sieve_x = 0;

        /**
         * m values that weren't composite from sieve
         * these values will be tested and any primes will be removed from testing_m
         */
        std::atomic<uint8_t> sieves_ready{0};
        vector<std::pair<uint32_t, vector<uint32_t>>> next_sieves;


        /** sieve_mtx must be held while calling all methods*/
        void setup_sieve_data(bool stop_after);
        void increment_X();
        bool try_set_testing_data(TestData &testing);
        void push_to_overflow_and_increment_M_range();

    private:
        vector<bool> is_m_coprime; // Needed for is_coprime_and_valid_m cache.
        vector<uint32_t> K_primes;
        vector<uint32_t> D_primes;

        void is_coprime_and_valid_m();
};


class GPUBatch {
    public:
        enum State : uint8_t { WAITING, EMPTY, RUNNING, DONE };
        std::atomic<State> state = WAITING;

        GpuStatsCounters stats;

        // Used in debugging Batch Timing.
        time_point<high_resolution_clock> fill_start;
        time_point<high_resolution_clock> fill_end;
        time_point<high_resolution_clock> gpu_start;
        time_point<high_resolution_clock> gpu_end;
        time_point<high_resolution_clock> results_start;
        time_point<high_resolution_clock> results_end;

        // current index.
        size_t i;

        // testing 'm * K + x'
        uint32_t x;

        // number to check if prime
        vector<mpz_t*> z;
        // XXX: This is an ugly hack because you can't create mpz_t vector easily
        mpz_t *z_array;
        // m_i corresponding to z
        vector<uint32_t> m_i;

        // If z[i] should be tested
        vector<uint8_t>  active;
        // Result from GPU
        vector<int>  result;

        uint32_t primes_in_batch;

        explicit GPUBatch(size_t n) {
            elements = n;

            z_array = (mpz_t *) malloc(n * sizeof(mpz_t));
            for (size_t i = 0; i < n; i++) {
                mpz_init(z_array[i]);
                z.push_back(&z_array[i]);
            }

            m_i.resize(n, 0);
            active.resize(n, 0);
            result.resize(n, -1);
        }

        ~GPUBatch() {
            //cout << "~GPUBatch" << endl;
            for (size_t i = 0; i < elements; i++) {
                mpz_clear(z_array[i]);
            }
            free(z_array);
        }

        GPUBatch(const GPUBatch&) = delete;
        GPUBatch& operator=(const GPUBatch&) = delete;

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

        size_t elements;
};
