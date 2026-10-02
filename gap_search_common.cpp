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

#include "gap_search_common.h"

#include <atomic>
#include <cstdint>


/** Shared state between threads */
std::atomic<bool> is_running;
std::atomic<uint8_t> stop_queue{0};


// TODO get this from the correct place
const uint32_t GPU_BATCH_SIZE = 8192;


TestData::TestData(const struct Config config)
        : stats(high_resolution_clock::now()) {
    m_inc = config.m_inc;
    verbose = config.verbose;

    found_prime_m_i.resize((m_inc/2 + 31) / 32, 0);
    full_reset();
}

/** Should hold lock during */
void TestData::print_stats() {
    double total_t = duration<double>(high_resolution_clock::now() - stats.s_start_t).count();
    setlocale(LC_NUMERIC, "");
    printf("\nGPU Timings (%.0f seconds):\n", total_t);
    printf("\tm               : %'lu (%'u/sec)\n",
            stats.total_m, (uint32_t) (stats.total_m / total_t));
    printf("\tm processed     : %'lu (%'u/sec)\n",
            stats.tested_m, (uint32_t) (stats.tested_m / total_t));
    printf("\ttotal tests     : %'lu (%.1f%% prime) (%'u/sec)\n",
            gpu_stats.total_prp_tests,
            100.0 * gpu_stats.total_primes / gpu_stats.total_prp_tests,
            (uint32_t) (gpu_stats.total_prp_tests / total_t));
    printf("\ttotal batches   : %'lu (%.5f secs/batch)\n",
            gpu_stats.batches_run, total_t / gpu_stats.batches_run);
    printf("\toverflowed      : %'lu (%.1f%% of ranges)\n",
            stats.s_gap_out_of_sieve_next,
            100.0 * stats.s_gap_out_of_sieve_next / stats.tested_m);
    printf("\tbatch fill %%    : %.1f%% (%% fill), %.1f%% (%% partial batch)\n",
            100.0 * gpu_stats.total_prp_tests / gpu_stats.batches_run / GPU_BATCH_SIZE ,
            100.0 * gpu_stats.batches_partial / gpu_stats.batches_run
    );
    printf("\twaiting 4 sieve : %.1f seconds (%.1f%%) %lu count \n",
            gpu_stats.d_wait_not_active, 100 * gpu_stats.d_wait_not_active / total_t,
            gpu_stats.wait_not_active);
    printf("\t---------------------------------------\n");
    if (gpu_stats.d_loop > 1)
        printf("\tlooping         : %.1f seconds (%.1f%%)\n",
                gpu_stats.d_loop, 100 * gpu_stats.d_loop / total_t);
    if (gpu_stats.d_lock > 1)
        printf("\tlocking         : %.1f seconds (%.1f%%)\n",
                gpu_stats.d_lock, 100 * gpu_stats.d_lock / total_t);
    printf("\tfilling batches : %.1f seconds (%.1f%%)\n",
            gpu_stats.d_fill, 100 * gpu_stats.d_fill / total_t);
    printf("\trunning on gpu  : %.1f seconds (%.1f%%)\n",
            gpu_stats.d_run, 100 * gpu_stats.d_run / total_t);
    printf("\tmisc            : %.1f seconds (%.1f%%)\n",
            gpu_stats.d_misc, 100 * gpu_stats.d_misc / total_t);
    printf("\tresults         : %.1f seconds (%.1f%%)\n",
            gpu_stats.d_results, 100 * gpu_stats.d_results / total_t);
    printf("\twait done X     : %.1f seconds (%.1f%%)\n",
            gpu_stats.d_done_x, 100 * gpu_stats.d_done_x / total_t);
    printf("\twait done M     : %.1f seconds (%.1f%%)\n",
            gpu_stats.d_done_m, 100 * gpu_stats.d_done_m / total_t);
    printf("\n");
    setlocale(LC_NUMERIC, "C");
}

void TestData::lock() {
    while (flag.exchange(1) == 1) {
        flag.wait(1, std::memory_order_relaxed);
    }
}

void TestData::unlock() {
    assert(flag.load() == 1); // locked (because we own it)
    flag = 0;
    flag.notify_one();
}

void TestData::wait_for_state_and_lock(State desired) {
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

void TestData::full_reset() {
    std::fill(found_prime_m_i.begin(), found_prime_m_i.end(), 0);
    unknown_m_i.clear();
    test_i = 0;

    state = State::WAITING;
}
