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

#include "gap_linear_testing.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <deque>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

// pthread_setname_np
#include <pthread.h>

#include "gap_common.h"
#include "gap_stats.h"
#include "gpu_testing.h"
#include "xoroshiro128plus.h"


using std::vector;
using namespace std::chrono;


const vector<uint16_t> VALID_GAPS = {
    10624, 10648, 10688, 10690, 10852, 10878,
    10930, 10946, 10954, 10972, 11002, 11038,
    11060, 11062, 11068, 11080, 11092, 11098,
    11104, 11114, 11116, 11122, 11138, 11146,
    // And everything bigger
};


LinearTestData::LinearTestData(const struct Config config)
        : stats(high_resolution_clock::now()) {
    verbose = config.verbose;

    offset = config.m_start;
    length = config.m_inc;

    init_K(config, test_k);

    // TODO number of batches from somewhere.
    ranges.resize(3 * GPU_BATCH_SIZE);

    composites.resize((length/2 + 31) / 32, 0);
    reset();
}

/** Should hold lock during */
void LinearTestData::print_stats() {
    double total_t = duration<double>(high_resolution_clock::now() - stats.s_start_t).count();
    setlocale(LC_NUMERIC, "");
    printf("\nGPU Timings (%.0f seconds):\n", total_t);
    printf("\tm               : %'lu (%'u/sec)\n",
            stats.total_m, (uint32_t) (stats.total_m / total_t));
    printf("\ttotal tests     : %'lu (%.1f%% prime) (%'u/sec)\n",
            gpu_stats.total_prp_tests,
            100.0 * gpu_stats.total_primes / gpu_stats.total_prp_tests,
            (uint32_t) (gpu_stats.total_prp_tests / total_t));
    printf("\tm / test        : %.1f\n",
            1.0 * stats.total_m / gpu_stats.total_prp_tests);
    printf("\ttotal batches   : %'lu (%.5f secs/batch)\n",
            gpu_stats.batches_run, total_t / gpu_stats.batches_run);
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
    //TODO probably delete
    //printf("\twait done X     : %.1f seconds (%.1f%%)\n",
    //        gpu_stats.d_done_x, 100 * gpu_stats.d_done_x / total_t);
    //printf("\twait done M     : %.1f seconds (%.1f%%)\n",
    //        gpu_stats.d_done_m, 100 * gpu_stats.d_done_m / total_t);
    printf("\n");
    setlocale(LC_NUMERIC, "C");
}

void LinearTestData::lock() {
    while (flag.exchange(1) == 1) {
        flag.wait(1, std::memory_order_relaxed);
    }
}

void LinearTestData::unlock() {
    assert(flag.load() == 1); // locked (because we own it)
    flag = 0;
    flag.notify_one();
}

void LinearTestData::wait_for_state_and_lock(State desired) {
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

void LinearTestData::reset() {
    std::fill(composites.begin(), composites.end(), 0);
    state = State::WAITING;
}

void LinearTestData::setup_ranges() {
    const uint64_t N = 3 * GPU_BATCH_SIZE;
    assert( ranges.size() ==  N );

    const auto B = length;

    for (uint64_t i = 0; i < N; i++) {
        auto& r = ranges[i];
        r.range_start = (i * B) / N;
        r.range_start += (r.range_start & 1);
        r.range_end = ((i+1) * B) / N;
        r.range_end += (r.range_end & 1);
        r.start = r.range_start;
        r.state = LinearRange::NEW;
    }

}

uint32_t process_finished_batch(LinearTestData &test_data, GPUBatch& batch) {
    test_data.lock();
    assert(batch.x == test_data.offset);
    assert(batch.capacity() == GPU_BATCH_SIZE);

    uint32_t found = 0;
    uint32_t test_i = 0;
    for (size_t i = 0; i < GPU_BATCH_SIZE; i++) {
        if (!batch.active[i]) {
            break;
        }
        // Verify GPU really did write the result
        assert (batch.result[i] == 0 || batch.result[i] == 1);

        uint32_t r_i = batch.m_i[i];
        auto &r = test_data.ranges[r_i];
        r.in_gpu_batch = 0;

        if (batch.result[i]) {
            found++;
            r.state = LinearRange::PRIME;
            test_i = i;
        }
    }

    test_data.unlock();

    // Spot check roughly one in a million.
    if (found > 0 && (rng_next() & 63) == 0) {
        // Spot check
        assert( mpz_probab_prime_p( batch.z_array[test_i], 20 ) );
    }

    return found;
}

/** test_data.lock should be held during call. */
inline void fill_batch(
        uint32_t runner_i,
        LinearTestData &test_data,
        GPUBatch& batch, const uint64_t offset) {
    assert( batch.state == GPUBatch::EMPTY );
    assert( batch.capacity() == GPU_BATCH_SIZE );

    // Grap some entries from each item in M

    batch.i = 0;
    batch.x = offset;
    // Turn off all entries in batch
    std::fill_n(batch.active.begin(), GPU_BATCH_SIZE, false);
    // Mark all results as invalid
    std::fill_n(batch.result.begin(), GPU_BATCH_SIZE, -1);

    {
        assert( test_data.state == LinearTestData::ACTIVE );
        uint32_t gpu_i = batch.i;  // [GPU] batch index
        // TODO jump by BATCH_SIZE if in_gpu_batch
        size_t r_i = 0;
        for (; r_i < test_data.ranges.size() && gpu_i < GPU_BATCH_SIZE; r_i++) {
            auto &r = test_data.ranges[r_i];
            if (r.in_gpu_batch || r.state == LinearRange::DONE)
                continue;

            r.in_gpu_batch = true;
            batch.active[gpu_i] = true;
            batch.m_i[gpu_i] = r_i;
            gpu_i++;
        }

        batch.i = gpu_i;
    }

    assert( batch.i <= GPU_BATCH_SIZE);

    if (test_data.verbose >= 4) {
        printf("\t\tFilled Batch(%u) | offset=%lu\n", runner_i, offset);
    }

    // Batches should be full unless lots of overflowed results.
    if (test_data.verbose >= 4 && batch.i > 0 && batch.i < GPU_BATCH_SIZE) {
        printf("Partial load @ offset=%lu -> %lu/%lu\n",
            offset, batch.i, GPU_BATCH_SIZE);
    }
}

/**
 * Starts a CPU thread that handles launching CUDA kernels for primality tests.
 * Multiple of these threads exist, one for each GPUBatch.
 * communicates with testing_thread via batch (GPUBatch)
 */
void run_gpu_thread(int runner_num, int verbose,
                    LinearTestData &test_data, GPUBatch& batch,
                    const mpz_t &K_in
                    ) {
    try {
        {
            std::string name = std::format("GPU({})", runner_num);
            pthread_setname_np(pthread_self(), name.c_str());
            std::ignore = nice(-1);
        }

        mpz_t K;
        mpz_init_set(K, K_in);

        bool just_paused = true;
        bool new_offset = true;
        uint64_t last_offset = 0;
        batch.results_end = high_resolution_clock::now();

        GPURunner runner{};

        size_t processed_batches = 0;
        while (is_running && stop_queue <= 1) {
            if (batch.state != GPUBatch::EMPTY) {
                batch.wait_for_state_and_lock(GPUBatch::EMPTY);
                batch.unlock();
            }
            if (!is_running || stop_queue > 1) {
                break;
            }

            assert( batch.state == GPUBatch::EMPTY );
            auto last_result_end = batch.results_end;

            { // Fill Batch logic
                batch.lock_start = high_resolution_clock::now();
                test_data.lock();

                batch.fill_start = high_resolution_clock::now();
                const auto offset = test_data.offset;
                fill_batch(runner_num, test_data, batch, offset);
                batch.fill_end = high_resolution_clock::now();

                if (batch.i == 0) {
                    batch.state = GPUBatch::WAITING;
                    just_paused = true;
                    // batch.unlock()
                    test_data.active_batches -= 1;
                    if (test_data.active_batches == 0) {
                        assert( test_data.running_batches == 0 );
                        test_data.state = LinearTestData::DONE;
                        test_data.state.notify_all();
                    }
                    test_data.unlock();
                    continue;
                } else {
                    test_data.running_batches += 1;
                    batch.state = GPUBatch::RUNNING;
                }
                test_data.unlock();

                uint64_t T = 0;

                // TODO refactor to a function probably
                // Batch was filled, test_data unlocked now handle updating ranges.
                for (uint32_t i = 0; i < batch.i; i++) {
                    auto &r = test_data.ranges[batch.m_i[i]];

                    if (r.state == LinearRange::PRIME) {
                        // if big gap -> Print
                        uint32_t gap = r.current - r.start + 1;
                        if (gap > 6000) {
                            printf("[%u] Found a big gap at (%lu, %lu) = %u\n",
                                    batch.m_i[i], offset + r.start, offset + r.current, gap);
                        }
                        r.start = r.current;
                        r.state = LinearRange::NEW;
                        if (r.start >= r.range_end) {
                            // TODO maybe shouldn't queue this or something, but rare so IDK
                            r.state = LinearRange::DONE;
                            // TODO maybe need mtx?
                            test_data.active_ranges = test_data.active_ranges - 1;
                        }
                    }

                    if (r.state == LinearRange::NEW) {
                        // Look at bits defined by gaps in VALID_GAPS
                        // Set jump to the first gap that is unknown (e.g. not composite)
                        // Set current = jump
                        // set PHASE1
                        r.current = r.start + VALID_GAPS.back() + 2;
                        if (r.current >= r.range_end) {
                            r.state = LinearRange::DONE;
                            test_data.active_ranges = test_data.active_ranges - 1;
                        } else {
                            for (const auto g : VALID_GAPS) {
                                uint64_t s = r.start + g;
                                assert( s < r.range_end );
                                uint64_t t = s >> 1;
                                if (!(test_data.composites[t >> 5] & (1 << (t & 31)))) {
                                    r.current = g;
                                    break;
                                }
                            }
                            r.state = LinearRange::PHASE1;
                        }
                    }
                    if (r.state == LinearRange::PHASE1) {
                        // Find previous unknown before current
                        // if no value before start
                        //      set state = PHASE2
                        //      set current = jump-1
                        // else
                        //       set T = current = ^
                        uint64_t s = r.current;
                        while (s > r.start) {
                            s -= 2;
                            uint64_t t = s >> 1;
                            if (!(test_data.composites[t >> 5] & (1 << (t & 31)))) {
                                break;
                            }
                        }
                        if (s > r.start) {
                            r.state = LinearRange::PHASE2;
                            r.current = r.jump;
                        } else {
                            r.current = s;
                        }
                    }
                    if (r.state == LinearRange::PHASE2) {
                        uint64_t s = r.current;
                        while (s < r.range_end) {
                            // This can probe at range_end
                            s += 2;
                            uint64_t t = s >> 1;
                            if (!(test_data.composites[t >> 5] & (1 << (t & 31)))) {
                                break;
                            }
                        }
                        if (s == r.range_end) {
                            printf("Gap >%lu that escaped range!\n", s - r.start);
                            r.state = LinearRange::DONE;
                        } else {
                            r.current = s;
                            T = s;
                        }
                        // Find next unknown after current
                        // set T = current = ^
                    }

                    if (false && i == 0) {
                        const auto &r = test_data.ranges[batch.m_i[i]];
                        printf("\t\tr[0] = {[%lu, %lu) | %u -> %lu %lu -> %lu\n",
                                r.range_start, r.range_end,
                                r.state,
                                r.start, r.current, T);

                    }

                    mpz_add_ui(*batch.z[i], K, offset + T);
                }

                new_offset = offset > last_offset;
                last_offset = offset;
            }

            // Verify all active items are all at the front of the batch.
            auto mid = batch.active.begin();
            std::advance(mid, batch.i);
            assert((uint32_t) std::count(batch.active.begin(), mid, 1) == batch.i);
            assert(std::count(mid,   batch.active.end(), 1) == 0);
            batch.gpu_start = high_resolution_clock::now();

            // Could batch.unlock(), no need.
            if (verbose >= 4)
                printf("\tGPU(%d): Starting batch %lu\n", runner_num, processed_batches);

            // run batch on gpu and wait for results to be set
            runner.run(batch);

            if (verbose >= 4)
                printf("\tGPU(%d): Finished batch %lu\n", runner_num, processed_batches);

            processed_batches += 1;

            // if batch.unlock() above would need to batch.lock() here.
            {
                batch.gpu_end = high_resolution_clock::now();

                // Process Batch (grabs test_data.lock internally)
                batch.primes_in_batch = process_finished_batch(test_data, batch);
                batch.state = GPUBatch::DONE;

                batch.results_end = high_resolution_clock::now();
            }
            { // Stats
                batch.stats.total_prp_tests += batch.i;
                batch.stats.total_primes += batch.primes_in_batch;

                double t_wait = duration<double>(batch.lock_start - last_result_end).count();
                double t_lock = duration<double>(batch.fill_start - batch.lock_start).count();
                double t_fill = duration<double>(batch.fill_end - batch.fill_start).count();
                double t_misc = duration<double>(batch.gpu_start - batch.fill_end).count();
                double t_run = duration<double>(batch.gpu_end - batch.gpu_start).count();
                double t_results = duration<double>(batch.results_end - batch.gpu_end).count();

                batch.stats.batches_run += 1;
                batch.stats.batches_partial += (batch.i < GPU_BATCH_SIZE);
                if (just_paused) {
                    // Happens at the end of each X (as we wait for other batches to finish)
                    // AND for the first X of each m
                    //printf("Paused for %.4f seconds @ new_offset: %lu, m=%lu x=%u\n",
                    //        t_wait, new_offset, test_data.m_start, test_data.testing_x);
                    if (new_offset)
                        batch.stats.d_done_m += t_wait;
                    else
                        batch.stats.d_done_x += t_wait;
                    just_paused = false;
                } else {
                    batch.stats.d_loop += t_wait;
                }
                batch.stats.d_lock += t_lock;
                batch.stats.d_fill += t_fill;
                batch.stats.d_misc += t_misc;
                batch.stats.d_run += t_run;
                batch.stats.d_results += t_results;

                if (verbose >= 4) {
                    test_data.lock();
                    printf("\tbatch(%u-%lu): %u primes | "
                            "batch timing [%.5f last], %.5f, %.5f, %.5f, %.5f, %.5f"
                            "%u running %u active\n",
                            runner_num, batch.stats.batches_run,
                            batch.primes_in_batch,
                            t_wait, t_lock, t_fill, t_misc, t_run, t_results,
                            test_data.running_batches.load() - 1, // -1 for us.
                            test_data.active_ranges.load()
                    );
                    test_data.unlock();
                }
            }

            batch.state = GPUBatch::EMPTY;
            // batch.unlock();
            // TODO is this safe?
            test_data.running_batches -= 1;
        }

        mpz_clear(K);

        if (verbose >= 2) {
            usleep(runner_num * 10'000); // i * 10ms
            printf("GPU(%d): Processed %'ld batches\n", runner_num, processed_batches);
        }

        if (!is_running) {
            // Signal to testing_thread it's time to be done
            test_data.state = LinearTestData::DONE;
            test_data.state.notify_all();
        }

    } catch (const std::exception &e) {
        cout << "ERROR in run_gpu_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}
