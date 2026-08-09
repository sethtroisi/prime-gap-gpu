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
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unistd.h>
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

// From 701# I believe.
// 1M -> 2000/second
// 200K -> 4000/second
const size_t CPU_SIEVE_LIMIT = 2'000;

//const size_t COMBINED_SIEVE_THREADS = 8;

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
         * NEW -> FIRST_SIEVE -> ACTIVE -> FINAL -> DONE
         * FIRST_SIEVE => Running the first sieve
         * FINAL => Don't sieve any more, just finish outstanding prime tests.
         */
        enum State { NEW, FIRST_SIEVE, ACTIVE, FINAL, DONE };
        State state = NEW;

        struct Config config;

        size_t current_testing_x = 0;
        /**
         * m_i values without a prime
         * this list corresponds to numbers BEFORE current_testing_x
         */
        vector<uint32_t> active_m_i;
        /* m_i values that need to be tested at current_testing_x */
        vector<uint32_t> unknown_m_i;
        // all indexes < test_i have been queued by fill_batch
        size_t test_i = 0;

        /* m_i where (m * K + current_testing_x) is prime at current_testing_x */
        vector<uint32_t> newly_prime_m_i;

        /* BITSET of m_i where a prime has been found (at any X). */
        /* used in conjunction with newly_prime_m_i to prevent double testing */
        vector<uint8_t> found_prime_m_i;

        /**
         * m values that weren't composite from sieve
         * these values will be tested and any primes will be removed from testing_m
         */
        size_t current_sieve_x = 0;
        vector<uint32_t> next_tests_m_i;

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
        // m_i corresponding to z
        vector<uint32_t> m_i;

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

            m_i.resize(n, 0);
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

// Don't read from sieve_data without holding sieve_mtx
std::mutex sieve_mtx;
std::unique_ptr<SieveData> sieve_data;

std::mutex overflow_mtx;
std::condition_variable overflow_cv;
// deque (double ended queue) avoids a degenerate case of large gap getting stuck
// if this can't keep up. Try to avoid falling behind, but this is an extra safety.
deque<std::pair<uint64_t, uint32_t>> overflowed;

    /**
     * globals:
     *      overflow_m: queue<uint64_t>
     *          m values where GPU never found next_prime (e.g. at least medium merit)
     *      active_m_i: uint8_t[M_inc]
     *          read_only by sieve_thread
     *          marked off by gpu_thread
     *      sieve_state: state
     *      gpu_state: state
     *
     * coordinator_thread
     *      * when GPU goes to 'DONE' (when not many left)
     *          * read active m and add to overflow_m
     *          * update m_start
     *          * reset active_m_i
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
 * Starts a CPU thread that handles launching CUDA kernels for primality tests.
 * Multiple of these threads exist, one for each GPUBatch.
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
                if (verbose >= 4)
                    printf("\tGPU(%d): Starting batch %lu\n", runner_num, processed_batches);
                runner.run_test(batch.i, batch.z, batch.result);
                if (verbose >= 4)
                    printf("\tGPU(%d): Finished batch %lu\n", runner_num, processed_batches);
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

        // Some prework
        mpz_t K;
        init_K(sieve_data->config, K);
        std::vector<std::pair<uint32_t, uint32_t>> p_and_r;
        primesieve::iterator iter;
        uint64_t prime = iter.next_prime();
        assert (prime == 2);  // we skip 2 which is the oddest prime.
        for (prime = iter.next_prime(); prime < CPU_SIEVE_LIMIT; prime = iter.next_prime()) {
            if (prime < sieve_data->config.p && sieve_data->config.d % prime > 0)
                continue;
            const uint32_t base_r = mpz_fdiv_ui(K, prime);
            p_and_r.emplace_back((uint32_t) prime, base_r);
        }


        assert(sieve_data);
        vector<uint32_t> local_active_m_i;
        struct Config config;
        uint64_t last_sieved = 0;

        while (queue_new_work) {
            lock.lock();
            uint64_t current_x = sieve_data->current_sieve_x;
            const auto state = sieve_data->state;
            if (state == SieveData::State::FIRST_SIEVE) {
                if (config.m_start != sieve_data->config.m_start) {
                    if (config.verbose >= 3)
                        printf("Reset GPU Sieve to 0\n");
                    last_sieved = 0;
                    sieve_data->next_tests_m_i.clear();
                }
            }

            if ((state != SieveData::State::FIRST_SIEVE && state != SieveData::State::ACTIVE)
                    || current_x == 0) {
                lock.unlock();
                usleep(1'000); // 1ms
                continue;
            }
            if (current_x == last_sieved) {
                lock.unlock();
                usleep(2'000); // 2ms waiting for tests to consume next_tests_m_i
                continue;
            }

            assert(sieve_data->next_tests_m_i.size() == 0);
            local_active_m_i = sieve_data->active_m_i;
            config = sieve_data->config;
            lock.unlock();

            auto s_start_t = high_resolution_clock::now();

            // TODO: sieve these numbers on GPU
            auto M_end = config.m_start + config.m_inc;
            vector<uint32_t> unknowns_i;
            {
                // Center is NOT ODD
                for (const auto m_i : local_active_m_i) {
                    uint64_t m = config.m_start + m_i;
                    if ((m * 1 + current_x) % 2 == 0) {
                        continue;
                    }
                    bool any_factor = false;
                    for( const auto& [p, r] : p_and_r) {
                        if ((m * r + current_x) % p == 0) {
                            any_factor = true;
                            break;
                        }
                    }
                    if (!any_factor) {
                        unknowns_i.push_back(m_i);
                    }
                }
            }
            assert(!unknowns_i.empty());

            auto s_stop_t = high_resolution_clock::now();
            if ((config.verbose
                        + (current_x <= 2)
                        + (config.m_start <= 1'000'000)) >= 3) {
                printf("\tGPU Sieve (%ldM to %ldM) @X=%lu with %ld/%ld unknown/active took %.1f seconds \n",
                       config.m_start / 1'000'000, M_end / 1'000'000,
                       current_x, unknowns_i.size(), local_active_m_i.size(),
                       duration<double>(s_stop_t - s_start_t).count());
            }

            lock.lock();

            // Finalize range
            last_sieved = current_x;
            sieve_data->next_tests_m_i = unknowns_i;
            if (state == SieveData::State::FIRST_SIEVE) {
                // Mark as active after next_tests_m_i are set
                sieve_data->state = SieveData::State::ACTIVE;
            }
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
        mpz_t K, center, next_p, prev_p;
        mpz_init_set(K, K_in);
        mpz_init(center);
        mpz_init(next_p);
        mpz_init(prev_p);

        StatsCounters stats(high_resolution_clock::now());
        struct Config config = sieve_data->config;

        double K_log = calc_log_K(config);
        const float min_merit = config.min_merit;
        // See THEORY.md! Added const is small preference for doing less prev_p.
        const float MIN_MERIT_TO_CONTINUE = 2.6 + std::log2(min_merit * std::log(2) + 1);
        const float MIN_GAP_TO_CONTINUE =  MIN_MERIT_TO_CONTINUE * (K_log + log(config.m_inc));

        // TODO I think this is global later
        /*
        std::vector<std::pair<uint32_t, uint32_t>> p_and_r;
        primesieve::iterator iter;
        uint64_t prime = iter.next_prime();
        assert (prime == 2);  // we skip 2 which is the oddest prime.
        for (prime = iter.next_prime(); prime < CPU_SIEVE_LIMIT; prime = iter.next_prime()) {
            const uint32_t base_r = mpz_fdiv_ui(K, prime);
            p_and_r.emplace_back((uint32_t) prime, base_r);
        }
        */

        std::unique_lock<std::mutex> lock(overflow_mtx, std::defer_lock);
        size_t tested = 0;

        while (is_running) {
            // Important so that overflow_cv / unlock waits correctly
            if (!lock.owns_lock())
                lock.lock();
            // Lock IS NOT held while waiting.
            overflow_cv.wait(lock, []{ return overflowed.size() || !is_running; });

            while (is_running && overflowed.size()) {
                // process_results does most printing. This is to gauge overflow size.
                if (tested % 1'000 == 0 && overflowed.size() > 50'000) {
                    printf("\tCPU Sieve Queue: %lu open, %lu processed\n",
                            overflowed.size(), tested);
                }

                auto m_and_x = overflowed.front(); overflowed.pop_front();
                auto m = m_and_x.first;
                auto min_x = m_and_x.second;

                mpz_mul_ui(center, K, m);
                mpz_add_ui(next_p, center, min_x);
                mpz_nextprime(next_p, next_p);
                mpz_sub(next_p, next_p, center);
                uint64_t next_gap = mpz_get_ui(next_p);

                if (next_gap < MIN_GAP_TO_CONTINUE) {
                    double merit = next_gap / (K_log + log(m));
                    stats.process_results(config, m, 0, next_gap, 0, 1, merit);
                    stats.s_skips_after_one_side += 1;
                } else {
                    mpz_prevprime(prev_p, center);
                    mpz_sub(prev_p, center, prev_p);
                    uint64_t prev_gap = mpz_get_ui(prev_p);
                    uint64_t gap = prev_gap + next_gap;
                    double merit = gap / (K_log + log(m));
                    stats.process_results(config, m, prev_gap, next_gap, 1, 1, merit);

                    if (merit > min_merit) {
                        // Double check, we only performed a single round of rabin miller on many numbers.
                        mpz_sub_ui(prev_p, center, prev_gap);
                        mpz_nextprime(next_p, prev_p);
                        mpz_sub(next_p, next_p, prev_p);
                        uint64_t test_gap = mpz_get_ui(next_p);
                        if (test_gap != gap) {
                            printf("GAP MISMATCH! %lu vs %lu at %lu * %u# / %u - %lu "
                                    "(could always be from 1 round of miller-rabin)\n",
                                    test_gap, gap, m, config.p, config.d, prev_gap);
                        }

                        printf("%lu %.3f %lu * %u# / %u - %lu\n",
                                gap, merit, m, config.p, config.d, prev_gap);
                    }
                }
                // TODO also pass in final X and verify tmp2 > X

                tested += 1;
            }
        }

        cout << "\tOverflowed " << tested << " intervals" << endl;
        mpz_clear(K);
        mpz_clear(center);
        mpz_clear(next_p);
        mpz_clear(prev_p);
    } catch (const std::exception &e) {
        cout << "ERROR in run_overflow_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


uint32_t process_finished_batch(GPUBatch& batch) {
    vector<uint32_t> prime_m_i;
    for (size_t i = 0; i < GPU_BATCH_SIZE; i++) {
        if (!batch.active[i]) {
            continue;
        }
        // Verify GPU really did write the result
        assert (batch.result[i] == 0 || batch.result[i] == 1);

        if (batch.result[i]) {
            // Found prime!
            prime_m_i.push_back(batch.m_i[i]);
        }
    }
    for (auto m_i : prime_m_i)
        sieve_data->newly_prime_m_i.push_back(m_i);
    return prime_m_i.size();
}

/** sieve_mtx must be held while calling */
uint32_t find_next_coprime_x(uint32_t x) {
    uint64_t any_coprime;
    do {
        // Advance to next valid X  <=>  gcd(K, X) == 1
        x += 1;
        any_coprime = false;

        if (sieve_data->config.d % 2 == 0) {
            // Center is odd, m is odd -> x must be even
            if (x % 2 == 1) {
                any_coprime = true;
                continue;
            }
        }

        for (auto prime : get_sieve_primes(sieve_data->config.p)) {
            if (sieve_data->config.d % prime == 0)
                continue;
            if (x % prime == 0) {
                any_coprime = true;
                break;
            }
        }
    } while (any_coprime);

    return x;
}


/** sieve_mtx must be held while calling */
void setup_sieve_data() {
    // Verify stuff
    assert(sieve_data->state == SieveData::State::NEW);

    sieve_data->current_sieve_x = find_next_coprime_x(0);
    sieve_data->current_testing_x = sieve_data->current_sieve_x;

    sieve_data->active_m_i.clear();
    sieve_data->unknown_m_i.clear();
    sieve_data->test_i = 0;
    sieve_data->newly_prime_m_i.clear();
    sieve_data->found_prime_m_i.clear();
    sieve_data->next_tests_m_i.clear();

    sieve_data->found_prime_m_i.resize((sieve_data->config.m_inc + 7) / 8 + 1, 0);

    const auto &config = sieve_data->config;
    // TODO use is_coprime_and_valid_m
    for (uint64_t m_i = 0; m_i < config.m_inc; m_i++) {
        uint64_t m = config.m_start + m_i;
        // Check if divisibly by any factor of D
        if (gcd(m, config.d) == 1) {
            sieve_data->active_m_i.push_back(m_i);
        }
    }
    if (config.verbose + (config.m_start <= 1) >= 2)
        printf("\nSetup, starting at X=%lu with %ld/%ld active_m\n",
                sieve_data->current_sieve_x,
                sieve_data->active_m_i.size(), config.m_inc);

    sieve_data->state = SieveData::State::FIRST_SIEVE;
}

/** Remove any element of A where bit A is set in B. */
void remove_vector(vector<uint32_t> &A, const vector<uint8_t> &B) {
    size_t i = 0;
    for (uint32_t a : A) {
        if (B[a >> 3] & (1 << (a & 7)))
            continue;
        A[i++] = a;
    }
    A.resize(i);
}


/** sieve_mtx must be held while calling */
void increment_X() {
    // TODO verify test_i > active_m_i
    // TODO verify no outstanding prime tests
    // TODO verify next_tests_m_i has been set

    // asserting next sieve is already done.
    assert(sieve_data->current_sieve_x > sieve_data->current_testing_x);
    uint64_t jump = sieve_data->current_sieve_x - sieve_data->current_testing_x;
    sieve_data->current_testing_x = sieve_data->current_sieve_x;
    if (sieve_data->config.verbose >= 3 && sieve_data->current_testing_x % 300 <= 1) {
        printf("\nMoving to X=%ld\n", sieve_data->current_testing_x);
    }

    for (auto m_i : sieve_data->newly_prime_m_i) {
        sieve_data->found_prime_m_i[m_i >> 3] |= 1 << (m_i & 7);
    }
    remove_vector(sieve_data->active_m_i, sieve_data->found_prime_m_i);
    sieve_data->unknown_m_i.clear();
    sieve_data->test_i = 0;

    // If sieve has already finished, copy it over
    assert (sieve_data->current_sieve_x == sieve_data->current_testing_x );
    if (!sieve_data->next_tests_m_i.empty()) {
        sieve_data->unknown_m_i = sieve_data->next_tests_m_i;
        remove_vector(sieve_data->unknown_m_i, sieve_data->found_prime_m_i);
        sieve_data->next_tests_m_i.clear();
        sieve_data->current_sieve_x = find_next_coprime_x(sieve_data->current_sieve_x);
    }

    sieve_data->newly_prime_m_i.clear();
}


/** sieve_mtx must be held while calling */
void push_to_overflow_and_increment_M_range(StatsCounters& stats) {
    // TODO refactor this out of increment_X
    for (auto m_i : sieve_data->newly_prime_m_i) {
        sieve_data->found_prime_m_i[m_i >> 3] |= 1 << (m_i & 7);
    }
    remove_vector(sieve_data->active_m_i, sieve_data->found_prime_m_i);

    const auto& config = sieve_data->config;
    uint64_t m_start = config.m_start;
    uint64_t m_inc = config.m_inc;
    uint32_t min_X = sieve_data->current_testing_x + 1;
    if (config.verbose >= 2) {
        printf("\n\n\nMoving to M_start from %ld to %ld\n",
                m_start, m_start + m_inc);
        printf("\tQueueing %ld for overflow (X>%ld)\n",
                sieve_data->active_m_i.size(),
                sieve_data->current_testing_x);
    }

    std::lock_guard lock(overflow_mtx);
    for (uint32_t m_i : sieve_data->active_m_i) {
        overflowed.push_back({m_start + m_i, min_X});
    }
    stats.s_gap_out_of_sieve_next += sieve_data->active_m_i.size();
    stats.s_tests += 1;
    stats.possibly_print_stats(config, m_start + m_inc - 1, 0, 0);

    sieve_data->config.m_start += m_inc;
    sieve_data->state = SieveData::State::NEW;
    setup_sieve_data();

    // CPU sieving thread will start if unlocked and notified
    overflow_cv.notify_one();
}


/**
 * sieve_mtx must be held while calling
 */
void fill_batch(
        uint32_t batch_i,
        GPUBatch& batch,
        const mpz_t &K) {
    assert( batch.state == GPUBatch::State::EMPTY);

    // Grap some entries from each item in M
    batch.i = 0;
    // Turn off all entries in batch
    std::fill_n(batch.active.begin(), GPU_BATCH_SIZE, false);
    // Mark all results as invalid
    std::fill_n(batch.result.begin(), GPU_BATCH_SIZE, -1);

    uint64_t m_start = sieve_data->config.m_start;

    mpz_t t;
    mpz_init(t);
    uint64_t X;
    size_t j;
    uint32_t first_test_i = sieve_data->test_i;
    {
        assert(sieve_data->state == SieveData::State::ACTIVE ||
               sieve_data->state == SieveData::State::FINAL);
        X = sieve_data->current_testing_x;

        uint32_t gpu_i = batch.i;  // [GPU] batch index
        j = sieve_data->test_i;
        for (; j < sieve_data->unknown_m_i.size() && gpu_i < GPU_BATCH_SIZE; j++) {
            // Skip any element where a previous prime was found.
            // Happens when primes from last X weren't removed from active_m before sieve started.
            uint32_t m_i = sieve_data->unknown_m_i[j];
            if (sieve_data->found_prime_m_i[m_i >> 3] & (1 << (m_i & 7)))
                continue;

            uint64_t m = m_start + sieve_data->unknown_m_i[j];
            mpz_mul_ui(t, K, m);
            mpz_add_ui(*batch.z[gpu_i], t, X);
            //gmp_printf("%lu * K + %lu = %Zd\n", m, X, batch.z[gpu_i]);
            batch.active[gpu_i] = true;
            batch.m_i[gpu_i] = sieve_data->unknown_m_i[j];
            gpu_i++;
        }

        sieve_data->test_i = j;
        batch.i = gpu_i;
    }

    mpz_clear(t);

    assert( batch.i <= GPU_BATCH_SIZE);

    if (sieve_data->config.verbose >= 1 && first_test_i < sieve_data->test_i) {
        //printf("\t\tFilled Batch(%u) | X=%lu -> [%u, %lu) of %lu\n",
        //    batch_i, X,
        //    first_test_i, sieve_data->test_i, sieve_data->unknown_m_i.size());
    }

    // Batches should be full unless lots of overflowed results.
    if (batch.i > 0 && batch.i < GPU_BATCH_SIZE) {
        //printf("Partial load @ %lu -> %lu this batch: %lu/%lu\n",
        //    X, j, batch.i, GPU_BATCH_SIZE);
    }
}

void run_testing_thread(const struct Config og_config) {
    // gap / 2 up to 60 merit
    // TODO add back distance_counts
    //uint64_t distance_counts[10000] = {};

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

        const uint64_t count_valid_m = count_num_m(og_config.m_start, og_config.m_inc, D);

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

            const uint64_t M_inc = og_config.m_inc;
            assert(count_valid_m > 0 && count_valid_m <= M_inc);
            //printf("\nTesting ranges of %'ld ~ %'ld m per range.\n\n", M_inc, count_valid_m);
            setlocale(LC_NUMERIC, "C");

            //printf("\tStarting to create GPU Batches\n");
        }

        StatsCounters gpu_stats(high_resolution_clock::now());

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
        uint64_t running_batches = 0;

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

            // Check if we are in the correct state to fille
            {
                std::unique_lock<std::mutex> lock(sieve_mtx);
                const auto state = sieve_data->state;
                if ((state != SieveData::State::ACTIVE) && (state != SieveData::State::FINAL)) {
                    lock.unlock();
                    usleep(10'000); // 10ms
                    continue;
                }

                if (sieve_data->unknown_m_i.empty()) {
                    assert( sieve_data->current_testing_x == sieve_data->current_sieve_x );
                    if (sieve_data->next_tests_m_i.empty()) {
                        lock.unlock();
                        usleep(1'000); // 1ms
                        continue;
                    }
                    // TODO refactor to a method shared with setup.
                    sieve_data->unknown_m_i = sieve_data->next_tests_m_i;
                    sieve_data->next_tests_m_i.clear();
                    sieve_data->current_sieve_x = find_next_coprime_x(sieve_data->current_sieve_x);
                    sieve_data->test_i = 0;
                }

                for (size_t i = 0; i < GPU_BATCHES; i++) {
                    GPUBatch& batch = gpu_batches[i];
                    batch.mutex.lock();

                    if (batch.state != GPUBatch::State::EMPTY) {
                        batch.mutex.unlock();
                        continue;
                    }

                    batch.fill_start = high_resolution_clock::now();
                    fill_batch(i, batch, K);
                    batch.fill_end = high_resolution_clock::now();

                    if (batch.i == 0) {
                        // Leave as empty and unlock
                        batch.mutex.unlock();
                    } else {
                        open_gpu.push(i);
                        running_batches += 1;

                        // Mark as ready, unlock, notify gpu thread;
                        //printf("\tMarked GPUBatch(%lu) as READY\n", i);
                        batch.state = GPUBatch::State::READY;
                        batch.mutex.unlock();
                        batch.cv.notify_one();
                    }
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
                    if (batch.state != GPUBatch::State::DONE)
                        break;

                    batch.results_start = high_resolution_clock::now();
                    sieve_mtx.lock();
                    running_batches -= 1;
                    uint32_t primes_in_batch = process_finished_batch(batch);
                    SieveData *d = sieve_data.get();
                    gpu_stats.s_total_prp_tests += batch.i

                    //printf("\tGot Finished Batch(%d)=%u prime | %lu running, %lu/%lu\n",
                    //        i, primes_in_batch,
                    //        running_batches, d->test_i, d->unknown_m_i.size());

                    // if all batches finished then move to next set
                    if (running_batches == 0 && d->test_i && d->test_i == d->unknown_m_i.size()) {
                        if (sieve_data->unknown_m_i.size() < count_valid_m * .004) {
                            // Less than 1% of original left
                            push_to_overflow_and_increment_M_range(gpu_stats);
                        } else {
                            increment_X();
                        }
                    }
                    sieve_mtx.unlock();


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
                    printf("\nFASTER WITH `make gap_search_gpu BITS=%ld` (may require `make clean`)\n\n", bits);
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

    sieve_data = std::make_unique<SieveData>(config);
    // This has output that's nicer close to the top.
    std::thread testing_thread(run_testing_thread, config);
    usleep(50'000); // 50ms

    // Setup
    {
        auto s_start_t = high_resolution_clock::now();
        setup_sieve_data();
        auto s_stop_t = high_resolution_clock::now();
        printf("\tSetup took %.1f seconds\n",
               duration<double>(s_stop_t - s_start_t).count());
    }
    std::thread sieve_thread(run_sieve_thread);
    std::thread overflow_sieve_thread(run_overflow_thread, std::ref(K));

    // WHAT IS SIGNAL I'M DONE?
    while (is_running) {
        usleep(100'000); // 100ms
    }

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
