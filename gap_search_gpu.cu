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

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <queue>
#include <ranges>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// pthread_setname_np
#include <pthread.h>

#include <gmp.h>
#include <primesieve.hpp>

#include "gap_common.h"
#include "gap_test_common.h"
#include "sieve_small.h"
#include "miller_rabin.h"

using std::cout;
using std::cerr;
using std::endl;
using std::deque;
using std::vector;
using namespace std::chrono;

#ifdef GPU_BITS
const int BITS = GPU_BITS;
#else
const int BITS = 1024;
#endif

const int WINDOW_BITS = (BITS <= 1024) ? 5 : 6;
const int THREADS_PER_INSTANCE = (BITS <= 512) ? 4 : 8;

/**
 * GPU_BATCHES the number of simultanious batches to create & queue.
 * GPU_BATCH_SIZE is 2^n | best is between 4K and 16K.
 */
const size_t GPU_BATCHES = 2;
const size_t GPU_BATCH_SIZE = 4 * 1024;

/********** BENCHMARKING ***********/
// 701#
// GPU    - BATCH TPI -> PRP/second
// 1080Ti - 2K,   8 -> 250K
// 1080Ti - 4K,   8 -> 266K
// 1080Ti - 8K,   8 -> 282K
// 1080Ti - 16K,  8 ->

// Remember to set sm_80
// A100   - 4K,   8 -> 930K
// A100   - 8K,   8 -> 1050K
// A100   - 16K,  8 -> 1085K!

// 347# as high as 180K m/sec
// 1080Ti - 8K, 4  -> 1680K
// 1080Ti - 4K, 4  -> 1890K!
// 1080Ti - 2K, 4  -> 1330K

// A100   - 8K,   4 ->
// A100   - 16K,  4 -> 2210K (40% utilization)
// A100   - 32K,  4 ->

// 257# as high as 340K m/sec!
// 1080Ti - 2K,  4 -> 2500K!
// 1080Ti - 4K,  4 -> 3300K!
// 1080Ti - 8K,  4 -> 3060K!

// A100   - 16K, 4 ->

/********** BENCHMARKING ***********/


// Always use 1.
const int ROUNDS = 1;
// 20-60% extra for overflow is very reasonable.
const size_t QUEUE_SIZE = 140 * GPU_BATCHES * GPU_BATCH_SIZE / 100;

// From 701# I believe.
// 1M -> 2000/second
// 200K -> 4000/second
const size_t CPU_SIEVE_LIMIT = 200'000;

const size_t COMBINED_SIEVE_THREADS = 8;

//************************************************************************

void prime_gap_test(const struct Config config);


int main(int argc, char* argv[]) {
    Config config = Args::argparse(argc, argv, Args::Pr::SEARCH_GPU);

    if (config.valid == 0) {
        Args::show_usage(argv[0], Args::Pr::SEARCH_GPU);
        return 1;
    }

    if (config.verbose >= 2) {
        printf("Compiled with GMP %d.%d.%d\n",
            __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR, __GNU_MP_VERSION_PATCHLEVEL);
    }

    if (100'000 < config.max_prime && config.max_prime > 4'000'000'000) {
        printf("\tmax_prime(%'ld) should be between 100K and 4B\n", config.max_prime);
    }

    setlocale(LC_NUMERIC, "");
    if (config.verbose >= 0) {
        printf("\n");
        printf("Testing m * %d#/%d\n", config.p, config.d);
    }

    if (config.m_skip != 0) {
        cout << "Use m_start not mskip" << endl;
        return 1;
    }

    setlocale(LC_NUMERIC, "C");

    prime_gap_test(config);

    cout << argv[0] << " ended" << endl;
}


class SieveData {
    public:
        SieveData(const struct Config config): config(config) {};

        /**
         * CONFIGURED -> SIEVING -> TESTING -> FINAL -> DONE
         * FINAL => GPUSieve all converted to DataM.
         *      After any outstanding GPUBatch's are finished, open_m will be 0.
         *      Either:
         *          change back to CONFIGURED and running next set of X.
         *          or pushing everything to overflow and mark DONE.
         */
        enum State { NEW, ACTIVE, FINAL, DONE };
        State state = NEW;

        struct Config config;

        size_t current_testing_x = 0;
        /**
         * m values without a prime
         * this list corresponds to numbers BEFORE current_testing_x
         */
        vector<uint32_t> active_m;
        // all indexes < test_i have been queued by fill_batch
        size_t test_i = 0;

        /**
         * m where (m * K + current_testing_x) is prime
         */
        vector<uint32_t> newly_prime_m;

        /**
         * m values that weren't composite from sieve
         * these values will be tested and any primes will be removed from testing_m
         */
        // TODO this probably needs a mutex/lock/state
        size_t current_sieve_x = 0;
        vector<uint32_t> next_tests_m;


        // TODO stats about number of tests performed...
};


class GPUBatch {
    public:
        enum State : uint8_t { EMPTY, READY, DONE };
        std::atomic<State> state = EMPTY;

        // Used in debugging Batch Timing.
        time_point<high_resolution_clock> fill_start;
        time_point<high_resolution_clock> fill_end;
        time_point<high_resolution_clock> gpu_start;
        time_point<high_resolution_clock> gpu_end;
        time_point<high_resolution_clock> results_start;
        time_point<high_resolution_clock> results_end;

        // current index;
        size_t i;

        // number to check if prime
        vector<mpz_t*> z;
        // XXX: This is an ugly hack because you can't create mpz_t vector easily
        mpz_t *z_array;
        // m corresponding to z
        vector<uint64_t> m;

        // If z[i] should be tested
        vector<char>  active;
        // Result from GPU
        vector<int>  result;


        // For signaling, must be owned to change state.
        std::mutex mutex;
        std::condition_variable cv;

        explicit GPUBatch(size_t n) {
            elements = n;

            z_array = (mpz_t *) malloc(n * sizeof(mpz_t));
            for (size_t i = 0; i < n; i++) {
                mpz_init(z_array[i]);
                z.push_back(&z_array[i]);
            }

            active.resize(n, 0);
            result.resize(n, -1);
        }

        ~GPUBatch() {
            cout << "~GPUBatch" << endl;
            for (size_t i = 0; i < elements; i++) {
                mpz_clear(z_array[i]);
            }
            free(z_array);
        }

        GPUBatch(const GPUBatch&) = delete;
        GPUBatch& operator=(const GPUBatch&) = delete;

    private:
        size_t elements;
};


/** Shared state between threads */
std::atomic<bool> is_running;
std::atomic<bool> queue_new_work;

// Don't read from sieve_data or config without holding sieve_mtx
std::mutex sieve_mtx;
struct Config config;
std::unique_ptr<SieveData> sieve_data;

std::mutex overflow_mtx;
std::condition_variable overflow_cv;
// deque (double ended queue) avoids a degenerate case of large gap getting stuck
// if this can't keep up. Try to avoid falling behind, but this is an extra safety.
deque<uint64_t> overflowed;

    /**
     * globals:
     *      overflow_m: queue<uint64_t>
     *          m values where GPU never found next_prime (e.g. at least medium merit)
     *      active_m: uint8_t[M_inc]
     *          read_only by sieve_thread
     *          marked off by gpu_thread
     *      sieve_state: state
     *      gpu_state: state
     *
     * coordinator_thread
     *      * when GPU goes to 'DONE' (when not many left)
     *          * read active m and add to overflow_m
     *          * update m_start
     *          * reset active_m
     *
     * sieve_thread: launches sieve kernel per X
     *      * takes current list of active M
     *      * sets those as active in GPU
     *      * asks GPU to mark off composites
     *      * passes list of not-composite numbers (X, <array m>) to GPU
     *      * starts on next X
     *          * This will test 5-15% extra "active" m but that's fine
     *          * Someone (GPU thread?) removes them before starting prime testing
     *
     * gpu_thread: lauches prime testing kernel
     *      * owns the list of active M
     *      * takes list of (X, <array m>) and tests
     *      * marks some numbers as not active
     *      * keeps the list
     *
     * overflow_sieve_thread: handles largest m per block
     *
     */


/**
 * Starts a CPU thread that handles launching CUDA kernels for primality tests
 * communicates with testing_thread via batch (GPUBatch)
 */
void run_gpu_thread(int verbose, int runner_num, GPUBatch& batch) {
    try {
        {
            std::string name = "GPU(" + std::to_string(runner_num) + ")";
            pthread_setname_np(pthread_self(), name.c_str());
        }

        // TODO test changing cudaDeviceScheduleBlockingSync to cudaDeviceScheduleYield or cudaDeviceScheduleSpin
        typedef mr_params_t<THREADS_PER_INSTANCE, BITS, WINDOW_BITS> params;
        test_runner_t<params> runner(GPU_BATCH_SIZE, ROUNDS);

        size_t processed_batches = 0;
        std::unique_lock lock(batch.mutex, std::defer_lock);
        while (is_running) {
            lock.lock();
            if (batch.state != GPUBatch::State::READY) {
                batch.cv.wait(lock, [&] { return batch.state == GPUBatch::State::READY || !is_running; });
            }

            if (!is_running) break;

            assert(batch.state == GPUBatch::State::READY);
            // Active items are all at the front of the batch.
            auto mid = batch.active.begin();
            std::advance(mid, batch.i);
            assert(std::count(batch.active.begin(), mid, 1) == batch.i);
            assert(std::count(mid,   batch.active.end(), 1) == 0);
            batch.gpu_start = high_resolution_clock::now();
            lock.unlock();

            // Run batch on GPU and wait for results to be set
            if (1) {
                //printf("GPU(%d): Starting batch %lu\n", runner_num, processed_batches);
                runner.run_test(batch.i, batch.z, batch.result);
                //printf("GPU(%d): Finished batch %lu\n", runner_num, processed_batches);
            } else {
                // Return true for 1/10 results (helps not overflow sieve)
                for (size_t gpu_i = 0; gpu_i < GPU_BATCH_SIZE; gpu_i++) {
                    if (batch.active[gpu_i]) {
                        batch.result[gpu_i] = (std::rand() % 10) == 1;
                    }
                }
            }

            lock.lock();

            batch.gpu_end = high_resolution_clock::now();
            processed_batches += 1;
            batch.state = GPUBatch::State::DONE;
            // let CPU thread unlock when it recieves the signal.
            lock.unlock();
            batch.cv.notify_one();
        }

        if (verbose >= 1) {
            printf("GPU(%d): Processed %'ld batches\n", runner_num, processed_batches);
        }
    } catch (const std::exception &e) {
        cout << "ERROR in run_gpu_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}

void run_sieve_thread(void) {
    try {
        pthread_setname_np(pthread_self(), "SIEVE_THREAD");

        std::unique_lock<std::mutex> lock(sieve_mtx, std::defer_lock);

        assert(sieve_data); // No one has set it up yet
        vector<uint32_t> local_active_m;
        struct Config config;

        while (queue_new_work) {
            lock.lock();
            uint64_t current_x = sieve_data->current_sieve_x;
            if (current_x == 0) {
                lock.unlock();
                usleep(1'000'000); // 1,000ms
                continue;
            }

            assert(sieve_data->next_tests_m.size() == 0);
            local_active_m = sieve_data->active_m;
            config = sieve_data->config;
            lock.unlock();

            auto s_start_t = high_resolution_clock::now();

            // TODO: sieve these numbers on GPU
            auto M_end = config.m_start + config.m_inc;
            {
            }

            auto s_stop_t = high_resolution_clock::now();
            printf("\tGPU Sieve (%ldM to %ldM) @X=%lu with %ld active took %.1f seconds\n",
                   config.m_start / 1'000'000, M_end / 1'000'000,
                   current_x, local_active_m.size(),
                   duration<double>(s_stop_t - s_start_t).count());

            lock.lock();
            // TODO: Set to all unknowns from sieving
            sieve_data->next_tests_m = local_active_m;
            lock.unlock();
        }
    } catch (const std::exception &e) {
        cout << "ERROR in run_sieve_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}

void run_overflow_thread(const mpz_t &K_in) {
    try {
        pthread_setname_np(pthread_self(), "CPU_OVERFLOW");
        mpz_t K, tmp1, tmp2;
        mpz_init_set(K, K_in);
        mpz_init(tmp1);
        mpz_init(tmp2);

        // TODO I think this is global later
        std::vector<std::pair<uint32_t, uint32_t>> p_and_r;
        primesieve::iterator iter;
        uint64_t prime = iter.next_prime();
        assert (prime == 2);  // we skip 2 which is the oddest prime.
        for (prime = iter.next_prime(); prime < CPU_SIEVE_LIMIT; prime = iter.next_prime()) {
            const uint32_t base_r = mpz_fdiv_ui(K, prime);
            p_and_r.emplace_back((uint32_t) prime, base_r);
        }

        std::unique_lock<std::mutex> lock(overflow_mtx, std::defer_lock);
        size_t tested = 0;

        while (is_running) {
            // Important so that overflow_cv / unlock waits correctly
            if (!lock.owns_lock())
                lock.lock();
            // Lock IS NOT held while waiting.
            overflow_cv.wait(lock, []{ return overflowed.size() || !is_running; });

            while (is_running && overflowed.size()) {
                // 16K batch might dump 160-400 per batch.
                if (tested % 10'000 == 0 && overflowed.size() > 1000) {
                    printf("\tCPU Sieve Queue: %lu open, %lu processed\n",
                            overflowed.size(), tested);
                }

                auto m = overflowed.front(); overflowed.pop_front();

                mpz_mul_ui(tmp1, K, m);
                mpz_nextprime(tmp2, tmp1);
                mpz_sub(tmp2, tmp2, tmp1);

                gmp_printf("nextprime(%lu * K) = %lu * K + %Zd\n", m, m, tmp2);

                tested += 1;
            }
        }

        cout << "\tOverflowed " << tested << " intervals" << endl;
        mpz_clear(K);
    } catch (const std::exception &e) {
        cout << "ERROR in run_overflow_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


/*
size_t add_to_processing(
        const mpz_t &K,
        const uint32_t D,
        std::vector<std::unique_ptr<DataM>> &processing) {
    // Add the n smallest m range from sieveds to processing.

    SieveHolder *holder = nullptr;
    {
        // Only need the lock while reading state
        std::lock_guard lock(sieve_mtx);
        for (auto& test : sieveds) {
            if (test->state == SieveHolder::TESTING) {
                if (holder == nullptr || test->sieve->M_start < holder->sieve->M_start) {
                    holder = test.get();
                }
            }
        }
    }

    if (!holder) {
        // Nothing to add!
        return 0;
    }

    const GPUSieve& sieve = *(holder->sieve);
    const auto& coprime_X = sieve.coprime_X;
    // For all the bits set in m_unknown_deltas
    const int32_t offset = sieve.unknown_X0;


    size_t added = 0;

    for (auto& interval : processing) {
        if (interval.get() != nullptr) {
            continue;
        }

        // Update current_index (next_index during the loop)
        uint32_t index = holder->current_index;
        holder->current_index += 1;
        assert( index < sieve.m_inc.size() );

        const auto m_inc = sieve.m_inc[index];
        auto m_unknown_deltas = sieve.unknowns[index];

        uint64_t m = sieve.M_start + m_inc;
        assert( gcd(m, D) == 1 );

        added += 1;
        interval.reset(new DataM(m));
        interval->unknowns.reserve(32);

        while (m_unknown_deltas) {
            int32_t i = ffsl(m_unknown_deltas) - 1;
            assert( m_unknown_deltas & (1 << i) );
            m_unknown_deltas ^= 1 << i;
            interval->unknowns.push_back(coprime_X[offset + i]);
        }
        // TODO track the number of remaining unknowns?
        // Should this happen here or somewhere else?
        holder->sieve->unknowns[index] = 0;

        mpz_init(interval->center);
        mpz_mul_ui(interval->center, K, interval->m);

        interval->parent = holder;
        interval->mii = index;

        if (index+1 == sieve.m_inc.size()) {
            std::lock_guard lock(sieve_mtx);
            holder->state = SieveHolder::FINAL;
            printf("Set m_start: %lu as FINAL\n", sieve.M_start);
            break;
        }
    }

    //printf("Added %lu from m_start: %lu index is now %lu/%lu\n",
    //    added, sieve.M_start, holder->current_index, sieve.m_inc.size());

    // No one else will be competing for this field so save to write without lock.
    holder->open_m += added;

    return added;
}
*/


void process_finished_batch(GPUBatch& batch) {
    vector<uint32_t> prime_m;
    for (size_t i = 0; i < GPU_BATCH_SIZE; i++) {
        if (!batch.active[i]) {
            continue;
        }
        // Verify GPU really did write the result
        assert (batch.result[i] == 0 || batch.result[i] == 1);

        if (batch.result[i]) {
            // Found prime!
            prime_m.push_back(batch.m[i]);
        }
    }
    for (auto m : prime_m)
        sieve_data->newly_prime_m.push_back(m);
}

void increment_X() {
    // TODO verify test_i > active_m
    // TODO verify no outstanding prime tests
    // TODO verify next_tests_m has been set
    // TODO do I need to hold the lock or does my parent?


    sieve_data->current_testing_x += 1;
    printf("Moving to X=%ld\n", sieve_data->current_testing_x);

    size_t p_i = 0;
    uint32_t prime_m = sieve_data->newly_prime_m.size() ? sieve_data->newly_prime_m[p_i] : 0xFFFFFFFF;
    size_t i = 0;
    for (uint32_t m : sieve_data->active_m) {
        while (m >= prime_m) {
            p_i += 1;
            bool skip = (m == prime_m);
            prime_m = sieve_data->newly_prime_m.size() ? sieve_data->newly_prime_m[p_i] : 0xFFFFFFFF;
            if (skip);
                continue;
        }
        sieve_data->active_m[i++] = m;
    }
    sieve_data->active_m.resize(i);

    sieve_data->newly_prime_m.clear();

    sieve_data->current_sieve_x = sieve_data->current_testing_x + 1;
    sieve_data->next_tests_m.clear();
}


void increment_M_range() {
    // TODO do I need to hold the lock or does my parent?
    std::lock_guard lock(overflow_mtx);
    uint64_t m_start = config.m_start;
    for (uint32_t m : sieve_data->active_m) {
        overflowed.push_back(m_start + m);
    }

    config.m_start += config.m_inc;

    // CPU sieving thread will start if unlocked and notified
    overflow_cv.notify_one();
}


void fill_batch(
        GPUBatch& batch,
        const mpz_t &K,
        StatsCounters& stats) {
    assert( batch.state == GPUBatch::State::EMPTY);

    // Grap some entries from each item in M
    batch.i = 0;
    // Turn off all entries in batch
    std::fill_n(batch.active.begin(), GPU_BATCH_SIZE, false);
    // Mark all results as invalid
    std::fill_n(batch.result.begin(), GPU_BATCH_SIZE, -1);

    // TODO make sure this has been set by this time
    mpz_t t;
    mpz_init(t);
    uint64_t X;
    size_t j;

    {
        std::lock_guard lock(sieve_mtx);
        X = sieve_data->current_testing_x;

        uint32_t gpu_i = batch.i;  // [GPU] batch index
        j = sieve_data->test_i;
        for (; j < sieve_data->active_m.size() && gpu_i < GPU_BATCH_SIZE; j++, gpu_i++) {
            mpz_mul_ui(t, K, sieve_data->active_m[j]);
            mpz_add_ui(*batch.z[gpu_i], t, X);
            batch.active[gpu_i] = true;
        }

        sieve_data->test_i = j;
    }

    mpz_clear(t);

    assert( batch.i <= GPU_BATCH_SIZE);

    // Batches should be full unless lots of overflowed results.
    if (batch.i > 0 && batch.i < GPU_BATCH_SIZE) {
        printf("Partial load @ %lu -> %lu this batch: %lu/%lu\n",
            X, j, batch.i, GPU_BATCH_SIZE);
    }
}

void run_testing_thread(const struct Config og_config) {
    // gap / 2 up to 60 merit
    uint64_t distance_counts[10000] = {};

    try {
        pthread_setname_np(pthread_self(), "CREATE_BATCHES");
        cout << endl;

        // K is initialized in prob_prime_and_stats
        mpz_t K;
        double K_log = prob_prime_and_stats(og_config, K);
        //const uint64_t P = og_config.p;
        const uint64_t D = og_config.d;

        const float min_merit = og_config.min_merit;

        // See THEORY.md! Added const is small preference for doing less prev_p.
        const float MIN_MERIT_TO_CONTINUE = 2.6 + std::log2(min_merit * std::log(2) + 1);

        // Print Header info
        if (og_config.verbose >= 1) {
            setlocale(LC_NUMERIC, "");
            // ----- Merit / Sieve stats
            float m_log = log(og_config.m_inc);
                printf("Min Gap ~= %'d (for merit > %.1f)\n",
                    (int) (min_merit * (K_log + m_log)), min_merit);
                printf("Min Gap to continue ~= %'d (for merit = %.1f)\n",
                    (int) (MIN_MERIT_TO_CONTINUE * (K_log + m_log)),
                    MIN_MERIT_TO_CONTINUE);

            const uint64_t M_start = og_config.m_start;
            const uint64_t M_inc = og_config.m_inc;
            uint64_t valid_ms = count_num_m(M_start, M_inc, D);
            assert(valid_ms > 0 && valid_ms <= M_inc);
            printf("\nTesting ranges of %'ld ~ %'ld m per range.\n\n", M_inc, valid_ms);
            setlocale(LC_NUMERIC, "C");
        }

        printf("\tStarting to create GPU Batches\n");

        // Used for various stats
        StatsCounters stats(high_resolution_clock::now());

        /* Note: Uses a double batched system
         * C++ Thread is preparing batch_a (even more m), while GPU runs batch_b */
        std::array<GPUBatch, GPU_BATCHES> gpu_batches = {
            GPUBatch(GPU_BATCH_SIZE),
            GPUBatch(GPU_BATCH_SIZE),
            //GPUBatch(GPU_BATCH_SIZE),
        };

        std::thread gpu_threads[GPU_BATCHES];
        for(size_t i = 0; i < GPU_BATCHES; i++) {
            gpu_threads[i] = std::thread(run_gpu_thread, og_config.verbose, i, std::ref(gpu_batches[i]));
        }

        // Silly but that's what life is.
        std::queue<int> open_gpu;

        //auto s_start_t = high_resolution_clock::now();
        //auto s_batch0_t = s_start_t;
        //auto s_batch1_t = s_start_t;

        // Main loop
        while (is_running) {
            /**
             * Try to fill all batches
             * queue on gpu all ready batches
             * wait for result from the 1st batch sent
             */

            for (size_t i = 0; i < GPU_BATCHES; i++) {
                GPUBatch& batch = gpu_batches[i];
                batch.mutex.lock();

                if (batch.state != GPUBatch::State::EMPTY) {
                    batch.mutex.unlock();
                    continue;
                }

                batch.fill_start = high_resolution_clock::now();
                fill_batch(batch, K, stats);
                batch.fill_end = high_resolution_clock::now();

                if (batch.i == 0) {
                    // Leave as empty and unlock
                    batch.mutex.unlock();
                } else {
                    open_gpu.push(i);

                    // Mark as ready, unlock, notify gpu thread;
                    batch.state = GPUBatch::State::READY;
                    batch.mutex.unlock();
                    batch.cv.notify_one();
                }
            }

            // Wait for the next batch to be done.
            {
                if (open_gpu.empty()) {
                    // Why would this happen, empty and nothing to queue?
                    usleep(50'000); // 50ms
                } else {
                    int i = open_gpu.front();
                    open_gpu.pop();

                    GPUBatch& batch = gpu_batches[i];
                    // Wait for the batch to be Done (unless it's already done)
                    std::unique_lock<std::mutex> lock(batch.mutex);
                    if (batch.state != GPUBatch::State::DONE) {
                        batch.cv.wait(
                            lock, [&] { return batch.state == GPUBatch::State::DONE ||!is_running; });
                    }

                    assert( batch.state == GPUBatch::State::DONE || !is_running );
                    if (!is_running)
                        break;

                    batch.results_start = high_resolution_clock::now();
                    sieve_mtx.lock();
                    process_finished_batch(batch);
                    // TODO if all batches finished then move to next set
                    if (sieve_data->test_i && sieve_data->test_i > sieve_data->active_m.size())
                        increment_X();
                    sieve_mtx.unlock();


                    // TODO
                    // increment_M_range();

                    batch.results_end = high_resolution_clock::now();
                    //if (rand() % (1 * 1024) == 0) {
                    if (0) {
                        // TODO check if gpu times are the same.
                        // If so that means that they are running side by side which maybe isn't what we want.
                        printf("CPU: batch timing fill: %.4f, to gpu: %.4f, "
                                "gpu: %.4f, to cpu: %.4f, process: %.4f\n",
                               duration<double>(batch.fill_end - batch.fill_start).count(),
                               duration<double>(batch.gpu_start - batch.fill_end).count(),
                               duration<double>(batch.gpu_end - batch.gpu_start).count(),
                               duration<double>(batch.results_start - batch.gpu_end).count(),
                               duration<double>(batch.results_end - batch.results_start).count());
                    }

                    // Result batch to EMPTY
                    batch.state = GPUBatch::State::EMPTY;
                }
            }
        }

        // ----- cleanup
        {
            mpz_clear(K);
        }

        // Send notifies (TODO Why is this needed?)
        for (auto& gpu_batch : gpu_batches) {
            gpu_batch.cv.notify_all();
        }
        size_t i = 0;
        for (auto & gpu_thread : gpu_threads) {
            gpu_thread.join();
            cout << "\tbatch gpu thread(" << i << ") joined" << endl;
            i++;
        }
    } catch (const std::exception &e) {
        cout << "ERROR in testing_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


void coordinator_thread(struct Config global_config) {
    try {
        pthread_setname_np(pthread_self(), "COORDINATOR");

        /*
        size_t total_ranges = 0;
        std::unique_lock<std::mutex> lock(sieve_mtx, std::defer_lock);
        while (is_running && (queue_new_work || sieveds.size())) {
            usleep(1'000'000); // 1,000ms

            std::unique_ptr<SieveHolder> to_test = nullptr;
            lock.lock();

            process_any_finals();

            if (queue_new_work) {  // Add new configs to sieveds queue.
                int needed = 2;
                int finished = 0;
                for (auto& sieved : sieveds) {
                    if (sieved->state == SieveHolder::CONFIGURED) {
                        needed -= 1;
                    }
                    if (sieved->state == SieveHolder::SIEVING) {
                        needed -= 1;
                        finished += 1;
                    }
                    if (sieved->state == SieveHolder::TESTING) {
                        needed -= 1;
                        finished += 1;
                    }
                }
                if (finished == 0 && total_ranges > 2) {
                    printf("Currently no finished ranges!!!\n");
                }

                while (needed > 0) {
                    // Add new config to the queue
                    if (global_config.verbose >= 3) {
                        printf("\tQueued(%lu) %'lu to %'lu\n",
                                total_ranges,
                                global_config.m_start,
                                global_config.m_start + global_config.minc);
                    }

                    sieveds.emplace_back(std::make_unique<SieveHolder>(global_config));
                    needed -= 1;
                    total_ranges += 1;

                    global_config.m_start += global_config.minc;
                }
            }
            lock.unlock();
        }
        */
        cout << "\n\nCoordinator Done.\n" << endl;
    } catch (const std::exception &e) {
        cout << "ERROR in coordinator_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}



void signal_callback_handler(int) {
    if (queue_new_work) {
       cout << endl;
       cout << "Caught CTRL+C stopping, winding down work." << endl;
       cout << endl;
       queue_new_work = false;
    } else {
       cout << endl;
       cout << "Caught 2nd CTRL+C, exit(2) now." << endl;
       cout << endl;
       is_running = false;
       exit(2);
    }
}

void prime_gap_test(struct Config config) {
    // Setup test runner
    printf("\n");
    printf("BITS=%d\tWINDOW_BITS=%d\n", BITS, WINDOW_BITS);
    printf("PRP/BATCH=%ld\n", GPU_BATCH_SIZE);
    printf("THREADS/PRP=%d\n", THREADS_PER_INSTANCE);

    assert( GPU_BATCH_SIZE == 1024 || GPU_BATCH_SIZE == 2048 || GPU_BATCH_SIZE == 4096 ||
            GPU_BATCH_SIZE == 8192 || GPU_BATCH_SIZE ==16384 || GPU_BATCH_SIZE ==32768 );

    mpz_t K;
    {
        init_K(config, K);
        // +4 is just is personal safety blanket buffer.
        size_t N_bits = mpz_sizeinbase(K, 2) + log2(config.m_start + 100ul * config.m_inc) + 4;

        // P# roughly 349, 709, 1063, 1447
        for (size_t bits : {512, 1024, 1536, 2048, 3036, 4096}) {
            if (N_bits <= bits) {
                if (bits < BITS) {
                    printf("\nFASTER WITH `make gap_test_gpu BITS=%ld` (may require `make clean`)\n\n", bits);
                    exit(1);
                }
                break;
            }
        }
        assert( (N_bits + 1) < BITS ); // See last debug line.
        assert( BITS <= (1 << (2 * WINDOW_BITS)) );
    }


    is_running     = true;
    queue_new_work = true;

    // Setup CTRL+C catcher
    signal(SIGINT, signal_callback_handler);

    std::thread main_thread(coordinator_thread, config);
    std::thread sieve_thread(run_sieve_thread);
    std::thread testing_thread(run_testing_thread, config);
    std::thread overflow_sieve_thread(run_overflow_thread, std::ref(K));

    main_thread.join();
    cout << "Joining threads" << endl;

    // Tell other threads to quit
    {
        is_running = false;
        sieve_thread.join();
        cout << "\tsieve joined" << endl;
        testing_thread.join();
        cout << "\ttesting joined" << endl;

        overflow_cv.notify_all();  // wake up all overflow thread
        overflow_sieve_thread.join();
        cout << "\toverflow joined" << endl;
    }

    mpz_clear(K);
}
