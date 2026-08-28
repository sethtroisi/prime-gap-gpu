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
#include <memory>
#include <experimental/propagate_const>

#include <gmp.h>

#include "gap_search_gpu.h"
#include "gap_stats.h"


extern const size_t GPU_BATCHES;
extern const size_t GPU_BATCH_SIZE;


class GPUBatch {
    public:
        enum State : uint8_t { WAITING, EMPTY, RUNNING, DONE };
        std::atomic<State> state = WAITING;

        GpuStatsCounters stats;

        // Used in debugging Batch Timing.
        time_point<high_resolution_clock> lock_start;
        time_point<high_resolution_clock> fill_start;
        time_point<high_resolution_clock> fill_end;
        time_point<high_resolution_clock> gpu_start;
        time_point<high_resolution_clock> gpu_end;
        time_point<high_resolution_clock> results_end;
        bool just_paused = true;

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

        void lock()  {
            while (flag.exchange(1) == 1) {
                flag.wait(1, std::memory_order_relaxed);
            }
        }

        void unlock() {
            assert(flag.load() == 1); // locked (because we own it)
            flag = 0;
            flag.notify_one();
        }

        void wait_for_state_and_lock(State desired) {
            while (1) {
                lock();
                auto current = state.load();
                if (current == desired || !is_running) {
                    return;
                }
                unlock();
                // Wait for state change
                state.wait(current);
            }
        }


    private:
        // For signaling, must be owned to change state.
        /**
         * :wait(0) -> unlock
         * -> set to 1 to lock with a check?
         */
        std::atomic<int> flag;

        size_t elements;
};


class GPURunner
{
    public:
        GPURunner();
        ~GPURunner();

        void run(GPUBatch& batch);

    private:
        class GPURunnerImpl;
        // PImpl
        std::experimental::propagate_const<
            std::unique_ptr<GPURunnerImpl>> pImpl;
};


void run_gpu_thread(int runner_num, int verbose,
                    TestData &test_data, GPUBatch& batch,
                    const mpz_t &K_in);

void gpu_state_and_checks(const mpz_t &K_in, const uint64_t m_end);
