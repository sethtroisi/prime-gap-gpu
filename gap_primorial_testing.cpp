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

#include "gap_primorial_testing.h"

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
#include "overflow.h"
#include "xoroshiro128plus.h"


using std::vector;
using namespace std::chrono;


uint32_t process_finished_batch(TestData &test_data, GPUBatch& batch) {
    test_data.lock();
    assert(batch.x == test_data.testing_x);
    assert(batch.capacity() == GPU_BATCH_SIZE);

    uint32_t found = 0;
    uint32_t m_i = 0;
    for (size_t i = 0; i < GPU_BATCH_SIZE; i++) {
        if (!batch.active[i]) {
            break;
        }
        // Verify GPU really did write the result
        assert (batch.result[i] == 0 || batch.result[i] == 1);

        if (batch.result[i]) {
            found++;
            m_i = batch.m_i[i];
            test_data.add_found_prime_m_i(m_i);
        }
    }

    // Spot check roughly one in a million.
    if (found > 0 && (rng_next() & 63) == 0) {
        // Spot check
        overflow.push_to_queue(
                test_data.m_start + m_i, batch.x,
                Overflow::Type::SPOT_CHECK);
    }

    test_data.unlock();
    return found;
}

/** test_data.lock should be held during call. */
inline void fill_batch(
        uint32_t gpu_i,
        TestData &test_data,
        GPUBatch& batch,
        const uint32_t x) {
    assert( batch.state == GPUBatch::EMPTY );
    assert( batch.capacity() == GPU_BATCH_SIZE );

    // Grap some entries from each item in M

    batch.i = 0;
    batch.x = x;
    // Turn off all entries in batch
    std::fill_n(batch.active.begin(), GPU_BATCH_SIZE, false);
    // Mark all results as invalid
    std::fill_n(batch.result.begin(), GPU_BATCH_SIZE, -1);

    uint32_t first_test_i = test_data.test_i;
    {
        assert( test_data.state == TestData::ACTIVE );
        uint32_t gpu_i = batch.i;  // [GPU] batch index
        size_t j = test_data.test_i;
        for (; j < test_data.unknown_m_i.size() && gpu_i < GPU_BATCH_SIZE; j++) {
            // Skip any element where a previous prime was found.
            // Happens when primes from last X weren't removed from active_m before sieve started.
            uint32_t m_i = test_data.unknown_m_i[j];
            uint32_t index = m_i >> 1;

            // Should happen only with a few primes found in last X.
            if (test_data.found_prime_m_i[index >> 5] & (1 << (index & 31)))
                continue;

            batch.active[gpu_i] = true;
            batch.m_i[gpu_i] = m_i;
            gpu_i++;
        }

        test_data.test_i = j;
        batch.i = gpu_i;
    }

    assert( batch.i <= GPU_BATCH_SIZE);

    if (test_data.verbose >= 4) {
        printf("\t\tFilled Batch(%u) | X=%u -> [%u, %lu) of %lu\n",
            gpu_i, x,
            first_test_i, test_data.test_i, test_data.unknown_m_i.size());
    }

    // Batches should be full unless lots of overflowed results.
    if (test_data.verbose >= 4 && batch.i > 0 && batch.i < GPU_BATCH_SIZE) {
        printf("Partial load @ %u -> %lu this batch: %lu/%lu\n",
            x, test_data.test_i, batch.i, GPU_BATCH_SIZE);
    }
}

/**
 * Starts a CPU thread that handles launching CUDA kernels for primality tests.
 * Multiple of these threads exist, one for each GPUBatch.
 * communicates with testing_thread via batch (GPUBatch)
 */
void run_gpu_thread(int runner_num, int verbose,
                    TestData &test_data, GPUBatch& batch,
                    const mpz_t &K_in) {
    try {
        {
            std::string name = std::format("GPU({})", runner_num);
            pthread_setname_np(pthread_self(), name.c_str());
            std::ignore = nice(-1);
        }

        mpz_t K, t;
        mpz_init_set(K, K_in);
        mpz_init(t);

        bool just_paused = true;
        bool new_m = true;
        uint64_t last_m = 0;
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
                assert( !test_data.unknown_m_i.empty() );
                const uint64_t m_start = test_data.m_start;
                const uint64_t x = test_data.testing_x;

                batch.fill_start = high_resolution_clock::now();
                fill_batch(runner_num, test_data, batch, x);
                batch.fill_end = high_resolution_clock::now();

                if (batch.i == 0) {
                    // Can happen if other batch took all remaining numbers or
                    // If last prime was just recently found prime
                    assert( test_data.test_i == test_data.unknown_m_i.size() );
                    batch.state = GPUBatch::WAITING;
                    just_paused = true;
                    // batch.unlock()
                    test_data.active_batches -= 1;
                    if (test_data.active_batches == 0) {
                        assert( test_data.running_batches == 0 );
                        test_data.state = TestData::DONE;
                        test_data.state.notify_all();
                    }
                    test_data.unlock();
                    continue;
                } else {
                    test_data.running_batches += 1;
                    batch.state = GPUBatch::RUNNING;
                }
                test_data.unlock();

                // Batch was filled, test_data unlocked now handle mpz_mul
                // XXX: Could move into todo with seperate entry point miller_rabin
                // helps offload a lot of mults, could reduce from_mpz cost
                for (uint32_t i = 0; i < batch.i; i++) {
                    uint64_t m = m_start + batch.m_i[i];
                    mpz_mul_ui(t, K, m);
                    mpz_add_ui(*batch.z[i], t, x);
                }

                new_m = m_start > last_m;
                last_m = m_start;
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
                    //printf("Paused for %.4f seconds @ new_m: %d, m=%lu x=%u\n",
                    //        t_wait, new_m, test_data.m_start, test_data.testing_x);
                    if (new_m)
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
                            "%u running %lu/%lu tested\n",
                            runner_num, batch.stats.batches_run,
                            batch.primes_in_batch,
                            t_wait, t_lock, t_fill, t_misc, t_run, t_results,
                            test_data.running_batches.load() - 1, // -1 for us.
                            test_data.test_i, test_data.unknown_m_i.size());
                    test_data.unlock();
                }
            }

            batch.state = GPUBatch::EMPTY;
            // batch.unlock();
            // TODO is this safe?
            test_data.running_batches -= 1;
        }

        mpz_clear(K);
        mpz_clear(t);;

        if (verbose >= 2) {
            usleep(runner_num * 10'000); // i * 10ms
            printf("GPU(%d): Processed %'ld batches\n", runner_num, processed_batches);
        }

        if (!is_running) {
            // Signal to testing_thread it's time to be done
            test_data.state = TestData::DONE;
            test_data.state.notify_all();
        }

    } catch (const std::exception &e) {
        cout << "ERROR in run_gpu_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}
