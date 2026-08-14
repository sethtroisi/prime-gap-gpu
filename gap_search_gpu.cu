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
#include <tuple>
#include <unistd.h>
#include <vector>

// pthread_setname_np
#include <pthread.h>

#include <gmp.h>
#include <primesieve.hpp>

#include "gap_common.h"
#include "gap_test_common.h"

//#define GPU_SIEVE
// GPU_SIEVE code is in 8ad9efb0

#ifdef GPU_SIEVE
#include "sieve_small.h"
#endif // GPU_SIEVE


#define GPU_TESTING

#ifdef GPU_TESTING
#include "miller_rabin.h"
#endif // GPU_TESTING

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
const size_t GPU_BATCHES = 3;
const size_t GPU_BATCH_SIZE = 4 * 1024;


// Always use 1.
const int ROUNDS = 1;

//const size_t COMBINED_SIEVE_THREADS = 8;

/**
 * Try to have completed this many sieve ahead of the GPU
 * Small extra cost of advance_X filtering for any recent primes
 * Big saving when X has few unknowns
 *      -> some ranges have 1/2 as many for some unknown (to seth) reason
 */
const size_t OPEN_SIEVES = 3;



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



//************************************************************************

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
    setlocale(LC_NUMERIC, "C");

    prime_gap_test(config);

    if (config.verbose >= 2)
        cout << argv[0] << " ended" << endl;
}


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
        /* m_i values that need to be tested at current_testing_x */
        vector<uint32_t> unknown_m_i;
        // all indexes < test_i have been queued by fill_batch
        size_t test_i = 0;

        /* BITSET of half of m_i where a prime has been found (at any X). */
        // TODO test impact of changing to uint32_t
        vector<uint8_t> found_prime_m_i;

        size_t sieve_x_i = 0;
        size_t current_sieve_x = 0;
        /**
         * m values that weren't composite from sieve
         * these values will be tested and any primes will be removed from testing_m
         */
        vector<std::pair<uint32_t, vector<uint32_t>>> next_sieves;

        /** sieve_mtx must be held while calling all methods*/
        void setup_sieve_data(bool stop_after);
        void increment_X();
        bool try_set_unknowns();
        void add_found_prime_m_i(const uint32_t m_i);
};


SieveData::SieveData(const struct Config config) {
    this->config = config;

    next_sieves.resize(OPEN_SIEVES);

    // Has roughly p bits -> find out to 10 merit way more than needed.
    {
        uint32_t stop_x = config.p * 10;

        vector<uint32_t> K_primes;
        for (auto prime : get_sieve_primes(config.p)) {
            if (config.d % prime != 0)
                K_primes.push_back(prime);
        }

        // Center is odd, m is odd -> x must be even
        assert(config.d % 2 == 0);

        for (uint32_t x = 2; x < stop_x ; x += 2) {
            uint64_t any_coprime;
            any_coprime = false;

            for (auto prime : K_primes) {
                if (x % prime == 0) {
                    any_coprime = true;
                    break;
                }
            }

            if (!any_coprime) {
                this->coprime_X.push_back(x);
            }
        }
    }
}

/** sieve_mtx must be held while calling */
void SieveData::setup_sieve_data(bool stop) {
    // Verify stuff
    assert(state == SieveData::State::NEW);

    sieve_x_i = 0;
    current_sieve_x = coprime_X[sieve_x_i];
    testing_x_i = 0;
    current_testing_x = coprime_X[testing_x_i];

    active_m_i.clear();
    unknown_m_i.clear();
    test_i = 0;
    found_prime_m_i.clear();

    for (auto& t : next_sieves) {
        t.first = 0;
        t.second.clear();
    }

    // Need to be able to write to config.minc >> 3
    found_prime_m_i.resize((config.m_inc/2) / 8 + 2, 0);

    if (stop)
        return;

    auto temp = is_coprime_and_valid_m(config);
    active_m_i.swap(temp.second);

    num_valid = active_m_i.size();
    if (config.verbose + (config.m_start <= 1) >= 3)
        printf("\nSetup, starting at X=%lu with %ld/%ld active_m\n",
                current_sieve_x, num_valid, config.m_inc);

    state = SieveData::State::FIRST_SIEVE;
}

/** Remove any element of A where bit A is set in B. */
void remove_vector(vector<uint32_t> &A, const vector<uint8_t> &B) {
    size_t i = 0;
    for (uint32_t a : A) {
        // TODO add comment and rename this slightly
        uint32_t index = a >> 1;
        if (B[index >> 3] & (1 << (index & 7)))
            continue;
        A[i++] = a;
    }
    A.resize(i);
}

inline void SieveData::add_found_prime_m_i(const uint32_t m_i) {
    assert(m_i < config.m_inc);
    // all m_i are even (see sieve) so shift down by 1
    uint32_t t = m_i >> 1;
    found_prime_m_i[t >> 3] |= 1 << (t & 7);
}

bool SieveData::try_set_unknowns() {
    assert(unknown_m_i.size() == 0);

    // Look for finished sieve to copy over.
    int32_t found_i = -1;
    for (uint32_t i = 0; i < OPEN_SIEVES; i++) {
        auto &[s_x, test_m_i] = next_sieves[i];
        if (s_x == current_testing_x) {
            s_x = 0; // TODO does this work?

            // Set testing data.
            test_i = 0;
            unknown_m_i.swap(test_m_i);
            // Do a second pass removing any primes that might have been found this round.
            remove_vector(unknown_m_i, found_prime_m_i);

            test_m_i.clear();
            found_i = i;
            break;
        }
    }

    if (found_i >= 0) {
        assert(next_sieves[found_i].first == 0);
        assert(next_sieves[found_i].second.empty());

        assert(unknown_m_i.size());
    }

    return found_i > 0;
}

/** sieve_mtx must be held while calling */
void SieveData::increment_X() {
    // TODO verify no outstanding prime tests
    assert(test_i == unknown_m_i.size());

    // Remove any found primes from active_m_i
    remove_vector(active_m_i, found_prime_m_i);
    unknown_m_i.clear();
    test_i = 0;

    // asserting next sieve is already done.
    testing_x_i++;
    current_testing_x = coprime_X[testing_x_i];
    if (config.verbose >= 3 && current_testing_x % 300 <= 1) {
        printf("\nMoving to X=%ld\n", current_testing_x);
    }

    // okay if it doesn't work
    try_set_unknowns();
}

// GLOBALS
/** Shared state between threads */
std::atomic<bool> is_running;
std::atomic<uint8_t> stop_queue(0);
/**
 * is_running = false
    stop immediately
 * stop_queue
    * 0: everything normal
    * 1: continue like normal till increment_m
    * 2: stop sieve & gpu_tester
         wait for overflow to finish
 */

// Don't read from sieve_data without holding sieve_mtx
std::mutex sieve_mtx;
std::unique_ptr<SieveData> sieve_data;

std::mutex overflow_mtx;
std::condition_variable overflow_cv;
// deque (double ended queue) avoids a degenerate case of large gap getting stuck
// if this can't keep up. Try to avoid falling behind, but this is an extra safety.
deque<std::pair<uint64_t, uint32_t>> overflowed;
// M * K + X which was marked prime by GPU. should be prime 100% of time.
deque<std::pair<uint64_t, uint32_t>> spot_check;

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
            //cout << "~GPUBatch" << endl;
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

#ifdef GPU_TESTING
        // TODO test changing cudaDeviceScheduleBlockingSync to cudaDeviceScheduleYield or cudaDeviceScheduleSpin
        typedef mr_params_t<THREADS_PER_INSTANCE, BITS, WINDOW_BITS> params;
        test_runner_t<params> runner(GPU_BATCH_SIZE, ROUNDS);
#endif // GPU_TESTING

        size_t processed_batches = 0;
        std::unique_lock lock(batch.mutex, std::defer_lock);
        while (is_running && stop_queue <= 1) {
            lock.lock();
            if (batch.state != GPUBatch::State::READY) {
                batch.cv.wait(lock, [&] { return batch.state == GPUBatch::State::READY
                        || !is_running || stop_queue > 1; });
            }

            if (!is_running || stop_queue > 1) break;

            assert(batch.state == GPUBatch::State::READY);
            // Verify all active items are all at the front of the batch.
            auto mid = batch.active.begin();
            std::advance(mid, batch.i);
            assert((uint32_t) std::count(batch.active.begin(), mid, 1) == batch.i);
            assert(std::count(mid,   batch.active.end(), 1) == 0);
            batch.gpu_start = high_resolution_clock::now();
            lock.unlock();

            // Run batch on GPU and wait for results to be set
#ifdef GPU_TESTING
            if (verbose >= 4)
                printf("\tGPU(%d): Starting batch %lu\n", runner_num, processed_batches);
            runner.run_test(batch.i, batch.z, batch.result);
            if (verbose >= 4)
                printf("\tGPU(%d): Finished batch %lu\n", runner_num, processed_batches);
#else
            // Return true for 1/10 results (helps not overflow sieve)
            for (size_t gpu_i = 0; gpu_i < GPU_BATCH_SIZE; gpu_i++) {
                if (batch.active[gpu_i]) {
                    batch.result[gpu_i] = (std::rand() % 10) == 1;
                }
            }
#endif // GPU_TESTING

            lock.lock();

            batch.gpu_end = high_resolution_clock::now();
            processed_batches += 1;
            batch.state = GPUBatch::State::DONE;
            // let CPU thread unlock when it recieves the signal.
            lock.unlock();
            batch.cv.notify_one();
        }

        if (verbose >= 2) {
            usleep(runner_num * 10'000); // i * 10ms
            printf("GPU(%d): Processed %'ld batches\n", runner_num, processed_batches);
        }
    } catch (const std::exception &e) {
        cout << "ERROR in run_gpu_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


/**
 * Return a^-1 mod p
 * a^-1 * a mod p = 1
 * assumes that gcd(a, p) = 1
 */
int32_t _invert(int32_t a, int32_t p) {
    // Use extended euclidean algorithm to get
    // a * x + p * y = 1
    // inverse of a is then x
    int x = 1, y = 0;
    int x1 = 0, y1 = 1, a1 = a, p1 = p;
    while (p1) {
        int32_t q = a1 / p1;

        int32_t t = x1;
        x1 = x - q * x1;
        x = t;

        t = y1;
        y1 = y - q * y1;
        y = t;

        t = p1;
        p1 = a1 - q * p1;
        a1 = t;
    }
    return x + p * (x < 0);
}


void run_sieve_thread(void) {
    try {
        pthread_setname_np(pthread_self(), "SIEVE_THREAD");
        std::ignore = nice(-2); // Increase priority a bit

        std::unique_lock<std::mutex> lock(sieve_mtx, std::defer_lock);

        auto s_thread_start_t = high_resolution_clock::now();

        // Some prework
        mpz_t K;
        struct Config config = sieve_data->config;
        init_K(config, K);
        std::vector<std::pair<uint32_t, uint32_t>> p_and_neg_inverse_k;
        std::vector<std::pair<uint32_t, uint32_t>> p_and_neg_inverse_k_small;
        std::vector<uint32_t> d_primes;
        uint64_t K_mod_d = mpz_fdiv_ui(K, config.d);
        {
            primesieve::iterator iter;
            uint64_t prime = iter.next_prime();
            assert (prime == 2);  // we skip 2 which is the oddest prime.
            for (prime = iter.next_prime(); prime < config.max_prime; prime = iter.next_prime()) {
                if (prime <= config.p) {
                    if (config.d % prime == 0)
                        d_primes.push_back(prime);
                    continue;
                }
                const uint32_t base_r = mpz_fdiv_ui(K, prime);
                assert( 0 < base_r && base_r < prime );
                const int32_t inv_K = _invert(base_r, prime);
                assert( 0 < inv_K && ((uint32_t) inv_K) < prime );
                assert( ((uint64_t) inv_K * base_r) % prime == 1 );
                // Never sieve less than this.
                if (prime < 100'000) {
                    p_and_neg_inverse_k_small.emplace_back((uint32_t) prime, prime - inv_K);
                } else {
                    p_and_neg_inverse_k.emplace_back((uint32_t) prime, prime - inv_K);
                }
            }
        }

#ifdef GPU_SIEVE
        GPUSieve gpu_sieve(config);
#endif // GPU_SIEVE

        assert ( mpz_odd_p(K) == true ); // Makes math below easier if true
        assert ( config.m_start % 2 == 0); // always start on even
        assert ( config.m_inc % 2 == 0); // all future m_start are even

        assert(sieve_data);
        const auto m_inc = config.m_inc;
        const auto M_INC_HALF = m_inc / 2;

        // Need to be able to write to composites[m_inc] as sentinel
        vector<uint8_t> composites(M_INC_HALF / 8 + 2, 0);
        const uint64_t composites_safe_idx = 8 * (composites.size() - 1);

        uint64_t D = config.d;
        assert( 4 * D % 8 == 0);
        vector<uint8_t> d_wheel(4*D, 0);
        vector<uint32_t> d_indexes;
        uint64_t total_m = 0;
        uint64_t total_runs = 0;
        uint64_t total_early_breaks = 0;
        uint64_t total_active = 0;
        uint64_t total_unknown = 0;
        uint64_t total_primes = 0;
        double total_time = 0;
        double finalize_time = 0;

        uint64_t max_p_i = p_and_neg_inverse_k.size();

        while (is_running && stop_queue <= 1) {
            lock.lock();
            uint32_t X = sieve_data->current_sieve_x;
            const auto state = sieve_data->state;
            if (state == SieveData::State::FIRST_SIEVE) {
                total_m += m_inc;
                max_p_i = p_and_neg_inverse_k.size();
                if (config.m_start != sieve_data->config.m_start) {
                    if (config.verbose >= 3)
                        printf("Reset GPU Sieve to 0\n");
                }
            }

            if ((state != SieveData::State::FIRST_SIEVE && state != SieveData::State::ACTIVE)
                    || X == 0) {
                lock.unlock();
                usleep(1'000); // 1ms
                continue;
            }

            // Check is if any OPEN_SIEVES is empty
            {
                uint32_t open_slots = 0;
                for (const auto& t : sieve_data->next_sieves) {
                    if (t.first == 0) {
                        open_slots++;
                        assert(t.second.size() == 0);
                    }
                }
                if (open_slots == 0) {
                    lock.unlock();
                    usleep(5'000); // 5ms waiting for tester to increment_X()
                    continue;
                }

                // tweak max_p_i if to many open_slots.
                if (open_slots > 1 && sieve_data->sieve_x_i > 10) {
                    max_p_i -= max_p_i / 12;
                }
            }

            config = sieve_data->config;

            lock.unlock();

            auto s_start_t = high_resolution_clock::now();
            double wheel_duration_t;
            const auto m_start = config.m_start;

            assert (D % 2 == 0);

            // M must be coprime with D  ->  M must be Odd
            // M_start -> even  ->  M_i = odd
            // M(odd) * K(odd) + X  ->  X must be even
            assert (X % 2 == 0);

            // Handle evens, maybe isn't needed?
            // TODO make sure always index zero.
            //for(uint32_t m_i = 0; m_i < m_inc; m_i += 2) {
            //    composites[m_i >> 3] |= 1 << (m_i & 7);
            //}
            assert(m_start % 2 == 0); // or fix the code
            // Used to mark of each even index by factor of 2, not needed because only odd indexes.
            //std::fill(composites.begin(), composites.end(), 0b1010101);
            std::fill(composites.begin(), composites.end(), 0);

            { // Handle all divisors of d at one time.
                auto s_wheel_t = high_resolution_clock::now();
                assert(D < config.m_inc / 100); // Do something different if not true.

                std::fill(d_wheel.begin(), d_wheel.end(), 0);

                for (uint32_t d : d_primes) {
                    if (d == 2) continue;
                    const uint64_t m_start_shift = m_start % d;
                    for (uint32_t m_i = 1; m_i < d_wheel.size(); m_i += 2) {
                        // check if d divides (m_start + m_i) * K + X
                        if (((m_start_shift + m_i) * K_mod_d + X) % d == 0) {
                            // mark all later multiples
                            for( uint32_t i = m_i >> 1; i < 8*d_wheel.size(); i += d ) {
                                d_wheel[i >> 3] |= 1 << (i & 7);
                            }
                            break;
                        }
                    }
                }

                // d_wheel = 8*config.d bits = multiple of d and 16, but 8 after removing half.
                for(uint32_t c_i = 0; c_i < composites.size(); ) {
                    size_t copy = std::min(composites.size() - c_i, d_wheel.size());
                    for (uint32_t j = 0; j < copy; j ++) {
                        composites[c_i + j] |= d_wheel[j];
                    }
                    c_i += copy;
                }
                wheel_duration_t = duration<double>(high_resolution_clock::now() - s_wheel_t).count();
            }

            uint32_t prime = 0;
            for( const auto& [p, neg_inv_K] : p_and_neg_inverse_k_small) {
                prime = p;
                // if ((m * K + X) % p == 0) {
                // if ((m * K) % p == -X) {
                // if ((m * K * inv_K) % p == -X * inv_K) {
                //uint64_t m_0 = (-X * inv_K) % prime;

                uint64_t m_start_shift = m_start % prime;
                m_start_shift = prime - m_start_shift;
                uint64_t mi_0 = (X * neg_inv_K + m_start_shift) % prime;

                // This requires K odd and m_start even (both checked above)
                // See 1ba32111 for more details.
                mi_0 += (mi_0 & 1) ? 0 : prime;
                mi_0 >>= 1; // Divide by 2 (even indexes aren't stored)

                uint32_t shift = prime;
                for (uint64_t t = mi_0; t < M_INC_HALF; t += shift) {
                    composites[t >> 3] |= 1 << (t & 7);
                }
            }

            /**
             * Optimizations
             *
             * if primes > M_INC_HALF can change loop to single unconditional set statement
             *      This is probably good but isn't relevant for the wide ranges currently being tested.
             */

            uint32_t neg_inv_K;
            if (X * config.max_prime < m_start) {
                for (uint32_t p_i = 0; p_i < max_p_i; p_i++) {
                    const auto& t = p_and_neg_inverse_k[p_i];
                    prime = t.first;
                    neg_inv_K = t.second;

                    // Avoids a 2nd '% prime' because m_start > X * neg_inv_K.
                    uint64_t mi_0 = (m_start - X * neg_inv_K) % prime;
                    mi_0 = (mi_0 == 0) ? 0 : prime - mi_0;

                    // This requires K odd and m_start even (both checked above)
                    // See 1ba32111 for more details.
                    mi_0 += (mi_0 & 1) ? 0 : prime;
                    mi_0 >>= 1; // Divide by 2 (even indexes aren't stored)

                    uint32_t shift = prime;
                    for (uint64_t t = mi_0; t < M_INC_HALF; t += shift) {
                        composites[t >> 3] |= 1 << (t & 7);
                    }
                }
            } else {
                for (uint32_t p_i = 0; p_i < max_p_i; p_i++) {
                    const auto& t = p_and_neg_inverse_k[p_i];
                    prime = t.first;
                    neg_inv_K = t.second;

                    uint64_t m_start_shift = m_start % prime;
                    m_start_shift = prime - m_start_shift;
                    uint64_t mi_0 = (X * neg_inv_K + m_start_shift) % prime;

                    // This requires K odd and m_start even (both checked above)
                    // See 1ba32111 for more details.
                    mi_0 += (mi_0 & 1) ? 0 : prime;

                    uint32_t shift = prime;
                    for (uint64_t t = mi_0; t < M_INC_HALF; t += shift) {
                        composites[t >> 3] |= 1 << (t & 7);
                    }
                }
            }

            auto s_stop_t = high_resolution_clock::now();
            double sieve_duration_t = duration<double>(s_stop_t - s_start_t).count();
            total_runs += 1;
            total_time += sieve_duration_t;
            total_primes += prime;

            if (config.verbose >= 3) {
                printf("\tsieve(X=%u) ended at %u after %.3f\n",
                        X, prime, sieve_duration_t);
            }

            lock.lock();

            if ( X != sieve_data->current_sieve_x) {
                //printf("At the end of sieve X=%u vs current=%u | state=%u, stop=%u\n",
                //        X, sieve_data->current_sieve_x, state
                assert( sieve_data->state == SieveData::State::FIRST_SIEVE || stop_queue >= 2);
                lock.unlock();
                continue;
            }

            double finalize_duration_t;
            uint32_t active_size = sieve_data->active_m_i.size();
            uint32_t unknowns_size;
            { // Finalize
                auto s_start_t = high_resolution_clock::now();

                vector<uint32_t> *tests = nullptr;
                for (auto& t : sieve_data->next_sieves) {
                    if (t.first == 0) {
                        t.first = X;
                        tests = &t.second;
                        break;
                    }
                }
                assert( tests != nullptr );
                assert( tests->empty() );

                if (! active_size ) {
                    printf("ERROR! No active_m_i at X=%u\n", X);
                } else {
                    assert( sieve_data->active_m_i.back() < m_inc );
                }

                // "most" (90%+) should be composite, so this should be < 12.5%
                for (auto m_i : sieve_data->active_m_i) {
                    //assert(m_i < m_inc);
                    //assert(m_i & 1 == 1); // M is odd, M start is even, m_i must be odd.
                    uint32_t t = m_i >> 1;
                    if (!(composites[t >> 3] & 1 << (t & 7)))
                        tests->push_back(m_i);
                }

                unknowns_size = tests->size();
                assert( active_size < 1'000 || unknowns_size > 0 );
                total_active += active_size;
                total_unknown += unknowns_size;

                auto s_stop_t = high_resolution_clock::now();
                finalize_duration_t = duration<double>(s_stop_t - s_start_t).count();
                finalize_time += finalize_duration_t;
                total_time += finalize_duration_t;

                // Finalize range
                sieve_data->sieve_x_i++;
                sieve_data->current_sieve_x = sieve_data->coprime_X[sieve_data->sieve_x_i];
                if (state == SieveData::State::FIRST_SIEVE) {
                    // Mark as active after next_sieves is set
                    sieve_data->state = SieveData::State::ACTIVE;
                }
            }

            // TODO revert
            if ((config.verbose + (X <= 2) + (config.m_start <= 1'000'000)) >= 3) {
                printf("\tGPU Sieve @X=%u with %u/%u (%.0f%%) unknown/active prime=%u"
                       " took %.3f (wheel: %.3f) + %.3f seconds\n",
                       X, unknowns_size, active_size,
                       100.0 * unknowns_size / active_size,
                       prime,
                       sieve_duration_t, wheel_duration_t, finalize_duration_t);
                if (0) {
                    printf("\tOpen sieves: ");
                    for (auto& t : sieve_data->next_sieves) {
                        printf("%u=%lu,", t.first, t.second.size());
                    }
                    printf("\n");
                }
            }

            lock.unlock();
        }

        mpz_clear(K);

        if (config.verbose >= 1 && total_runs > 0) {
            double total_s = duration<double>(high_resolution_clock::now() - s_thread_start_t).count();
            setlocale(LC_NUMERIC, "");
            printf("\nSIEVE Timings:\n");
            printf("\ttotal_m: %'lu (%'u/second) %.1f seconds\n",
                    total_m, (uint32_t) (total_m / total_s), total_s);
            printf("\tsieves: %lu (%.1f%% early exit)\n",
                    total_runs, 100.0 * total_early_breaks / total_runs);
            printf("\tavg prime: %'lu\n", total_primes / total_runs);
            printf("\tfinalize_time(%.1f%%): %.1f seconds (%.3f/sieve)\n",
                    100 * finalize_time / total_time, finalize_time, finalize_time / total_runs);
            printf("\ttotal_time: %.1f seconds (%.3f/sieve)\n",
                    total_time, total_time / total_runs);
            printf("\ttotal_active: %'lu, total_unknown: %'lu (%.2f%%)\n",
                    total_active, total_unknown, 100.0 * total_unknown / total_active);
            printf("\tactive / run: %'lu, unknown / run: %'lu\n",
                    total_active / total_runs, total_unknown / total_runs);
            printf("\n");
            setlocale(LC_NUMERIC, "C");
        }

    } catch (const std::exception &e) {
        cout << "ERROR in run_sieve_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


class TestingStats {
    public:
        std::atomic<uint64_t> tested = 0;
        std::atomic<uint64_t> skipped_prev = 0;
        std::atomic<uint64_t> tested_prev = 0;
        std::atomic<uint64_t> greater_than_min_merit = 0;
        std::atomic<uint64_t> mismatches = 0;

        std::atomic<uint64_t> spot_checked = 0;
};
TestingStats stats;

bool overflow_should_run() {
    return !is_running || stop_queue >= 2 || overflowed.size();
}

void run_cpu_overflow_thread(uint32_t i, const mpz_t &K_in) {
    try {
        {
            char name[16];
            sprintf(name, "CPU_OVERFLOW_%u\n", i);
            pthread_setname_np(pthread_self(), name);
            std::ignore = nice(+10); // Lower priority a bit
        }

        mpz_t K, center, next_p, prev_p;
        mpz_init_set(K, K_in);
        mpz_init(center);
        mpz_init(next_p);
        mpz_init(prev_p);

        //StatsCounters stats(high_resolution_clock::now());
        struct Config config = sieve_data->config;

        double K_log = calc_log_K(config);
        const float min_merit = config.min_merit;
        // See THEORY.md! Added const is small preference for doing less prev_p.
        const float MIN_MERIT_TO_CONTINUE = 2.6 + std::log2(min_merit * std::log(2) + 1);
        const float MIN_GAP_TO_CONTINUE =  MIN_MERIT_TO_CONTINUE * (K_log + log(config.m_inc));

        // 2-5x what comes in per batch
        const uint64_t overflow_too_much = config.m_inc * config.cpu_fraction;

        std::unique_lock<std::mutex> lock(overflow_mtx);
        while (is_running && (stop_queue < 2 || overflowed.size() > 0)) {
            assert(lock.owns_lock());
            // Lock IS NOT held while waiting.
            overflow_cv.wait(lock, overflow_should_run);

            while (is_running && (overflowed.size() || spot_check.size())) {
                assert(lock.owns_lock());

                // process_results does most printing. This is to gauge overflow size.
                if (stats.tested % 10'000 == 0 && overflowed.size() > overflow_too_much) {
                    printf("\tCPU Sieve Queue: %lu open, %lu processed\n",
                            overflowed.size(), stats.tested.load());
                }

                if (stop_queue > 0) {
                    uint32_t rem = overflowed.size();
                    bool is_power_print = false;
                    for (uint64_t p = 1000; p <= rem; p *= 10) {
                        is_power_print |= (rem == p) || (rem == 2*p) || (rem == 5*p);
                    }
                    if (is_power_print) {
                        printf("\tFinalizing(stage %d): %u open, %lu processed\n",
                            stop_queue.load(), rem, stats.tested.load());
                    }
                }

                if (!overflowed.size()) {
                    stats.spot_checked++;
                    assert(spot_check.size());
                    auto m_and_x = spot_check.front(); spot_check.pop_front();
                    mpz_mul_ui(center, K, m_and_x.first);
                    mpz_add_ui(next_p, center, m_and_x.second);
                    if (!mpz_probab_prime_p(next_p, 20)) {
                        printf("\n\n");
                        printf("%lu'th SPOT CHECK FAILED!\n", stats.spot_checked.load());
                        printf("%lu * %u# / %u + %u is not prime!\n",
                                m_and_x.first, config.p, config.d, m_and_x.second);
                        printf("\n\n");
                        exit(1);
                    }
                    continue;
                }

                auto m_and_x = overflowed.front(); overflowed.pop_front();
                lock.unlock();

                stats.tested++;

                auto m = m_and_x.first;
                auto min_x = m_and_x.second;

                mpz_mul_ui(center, K, m);
                mpz_add_ui(next_p, center, min_x);
                mpz_nextprime(next_p, next_p);
                mpz_sub(next_p, next_p, center);
                uint64_t next_gap = mpz_get_ui(next_p);

                if (next_gap < MIN_GAP_TO_CONTINUE) {
                    stats.skipped_prev++;
                } else {
                    stats.tested_prev++;
                    mpz_prevprime(prev_p, center);
                    mpz_sub(prev_p, center, prev_p);
                    uint64_t prev_gap = mpz_get_ui(prev_p);
                    uint64_t gap = prev_gap + next_gap;
                    double merit = gap / (K_log + log(m));

                    if (merit > min_merit) {
                        stats.greater_than_min_merit++;
                        // Double check, we only performed a single round of rabin miller on many numbers.
                        mpz_sub_ui(prev_p, center, prev_gap);
                        mpz_nextprime(next_p, prev_p);
                        mpz_sub(next_p, next_p, prev_p);
                        uint64_t test_gap = mpz_get_ui(next_p);
                        if (test_gap != gap) {
                            stats.mismatches++;
                            printf("\tGAP MISMATCH! %lu vs %lu at %lu * %u# / %u - %lu "
                                    "(probably because only 1 miller-rabin test)\n",
                                    test_gap, gap, m, config.p, config.d, prev_gap);
                            merit = test_gap / (K_log + log(m));
                        }

                        if (merit > min_merit) {
                            printf("%lu %.3f %lu * %u# / %u - %lu\n",
                                    gap, merit, m, config.p, config.d, prev_gap);
                        }
                    }
                }

                lock.lock();
            }
        }

        if (i == 0 && config.verbose >= 1) {
            printf("\nCPU OVERFLOW Timing:\n");
            printf("\ttotal tested: %lu\n", stats.tested.load());
            printf("\tspot checked: %lu\n", stats.spot_checked.load());
            printf("\tnext prime only: %lu, both sides: %lu\n",
                    stats.skipped_prev.load(), stats.tested_prev.load());
            uint32_t large = stats.mismatches + stats.greater_than_min_merit;
            if (large) {
                printf("\t> %.1f merit: %lu (%lu = %.1f%% bad next_prime)\n",
                        config.min_merit, stats.greater_than_min_merit.load(), stats.mismatches.load(),
                        100.0 * stats.mismatches.load() / large);
            }
            printf("\n");
        }

        if (config.verbose >= 3) {
            usleep(i * 10'000); // i * 10ms
            printf("\tCPU overflow(%u) done", i);
        }

        mpz_clear(K);
        mpz_clear(center);
        mpz_clear(next_p);
        mpz_clear(prev_p);
    } catch (const std::exception &e) {
        cout << "ERROR in run_cpu_overflow_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


/** sieve_mtx must be held while calling */
void push_to_overflow_and_increment_M_range(StatsCounters& stats) {
    remove_vector(sieve_data->active_m_i, sieve_data->found_prime_m_i);

    const auto& config = sieve_data->config;
    uint64_t m_start = config.m_start;
    uint64_t m_inc = config.m_inc;
    uint32_t min_X = sieve_data->current_testing_x + 1;
    if (config.verbose >= 2) {
        printf("\n\n\nMoving to M_start from %ld to %ld\n",
                m_start, m_start + m_inc);
        printf("\tQueueing %ld for overflow (X>%ld)",
                sieve_data->active_m_i.size(),
                sieve_data->current_testing_x);
        if (overflowed.size())
            printf(" (current: %lu)", overflowed.size());
        printf("\n");
    }

    std::lock_guard lock(overflow_mtx);
    for (uint32_t m_i : sieve_data->active_m_i) {
        overflowed.emplace_back(m_start + m_i, min_X);
    }
    stats.s_gap_out_of_sieve_next += sieve_data->active_m_i.size();
    stats.s_tests += sieve_data->num_valid;
    // TODO make this print occasionally with good formatting.
    //stats.possibly_print_stats("GPU", config);

    sieve_data->config.m_start += m_inc;
    sieve_data->state = SieveData::State::NEW;
    sieve_data->setup_sieve_data(stop_queue > 0);

    // CPU sieving thread will start if unlocked and notified
    overflow_cv.notify_all();
}



/** sieve_mtx must be held while calling */
uint32_t process_finished_batch(GPUBatch& batch) {

    uint32_t found = 0;
    uint32_t m_i = 0;
    for (size_t i = 0; i < GPU_BATCH_SIZE; i++) {
        if (!batch.active[i]) {
            continue;
        }
        // Verify GPU really did write the result
        assert (batch.result[i] == 0 || batch.result[i] == 1);

        if (batch.result[i]) {
            found++;
            m_i = batch.m_i[i];
            sieve_data->add_found_prime_m_i(m_i);
        }
    }
    assert(batch.x == sieve_data->current_testing_x);

    if (found > 0 && (rand() & 255) == 0) {
        // Spot check
        std::lock_guard lock(overflow_mtx);
        spot_check.emplace_back(sieve_data->config.m_start + m_i, batch.x);
    }
    return found;
}

/** sieve_mtx must be held while calling */
inline void fill_batch(
        uint32_t batch_i,
        GPUBatch& batch,
        const mpz_t &K,
        mpz_t &t,
        const uint32_t x) {
    assert( batch.state == GPUBatch::State::EMPTY);

    // Grap some entries from each item in M

    batch.i = 0;
    batch.x = x;
    // Turn off all entries in batch
    std::fill_n(batch.active.begin(), GPU_BATCH_SIZE, false);
    // Mark all results as invalid
    std::fill_n(batch.result.begin(), GPU_BATCH_SIZE, -1);

    uint64_t m_start = sieve_data->config.m_start;

    size_t j;
    uint32_t first_test_i = sieve_data->test_i;
    {
        assert(sieve_data->state == SieveData::State::ACTIVE ||
               sieve_data->state == SieveData::State::FINAL);
        uint32_t gpu_i = batch.i;  // [GPU] batch index
        j = sieve_data->test_i;
        for (; j < sieve_data->unknown_m_i.size() && gpu_i < GPU_BATCH_SIZE; j++) {
            // Skip any element where a previous prime was found.
            // Happens when primes from last X weren't removed from active_m before sieve started.
            uint32_t m_i = sieve_data->unknown_m_i[j];
            uint32_t index = m_i >> 1;
            if (sieve_data->found_prime_m_i[index >> 3] & (1 << (index & 7)))
                continue;

            uint64_t m = m_start + sieve_data->unknown_m_i[j];
            mpz_mul_ui(t, K, m);
            mpz_add_ui(*batch.z[gpu_i], t, x);
            batch.active[gpu_i] = true;
            batch.m_i[gpu_i] = sieve_data->unknown_m_i[j];
            gpu_i++;
        }

        sieve_data->test_i = j;
        batch.i = gpu_i;
    }

    assert( batch.i <= GPU_BATCH_SIZE);

    if (sieve_data->config.verbose >= 4 && first_test_i < sieve_data->test_i) {
        printf("\t\tFilled Batch(%u) | X=%u -> [%u, %lu) of %lu\n",
            batch_i, x,
            first_test_i, sieve_data->test_i, sieve_data->unknown_m_i.size());
    }

    // Batches should be full unless lots of overflowed results.
    if (sieve_data->config.verbose >= 4 && batch.i > 0 && batch.i < GPU_BATCH_SIZE) {
        printf("Partial load @ %u -> %lu this batch: %lu/%lu\n",
            x, j, batch.i, GPU_BATCH_SIZE);
    }
}

void run_testing_thread(const struct Config og_config) {
    // gap / 2 up to 60 merit
    // TODO add back distance_counts
    //uint64_t distance_counts[10000] = {};

    try {
        pthread_setname_np(pthread_self(), "CREATE_BATCHES");
        std::ignore = nice(-2); // Increase priority a bit
        cout << endl;


        // K is initialized in prob_prime_and_stats
        mpz_t K, t;
        mpz_init(t);
        double K_log = prob_prime_and_stats(og_config, K);
        //const uint64_t P = og_config.p;
        const uint64_t D = og_config.d;

        const float min_merit = og_config.min_merit;

        // See THEORY.md! Added const is small preference for doing less prev_p.
        const float MIN_MERIT_TO_CONTINUE = 2.6 + std::log2(min_merit * std::log(2) + 1);

        const uint64_t count_valid_m = count_num_m(og_config.m_start, og_config.m_inc, D);
        const uint64_t overflow_count = count_valid_m * sieve_data->config.cpu_fraction;

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
            printf("\nTesting ranges of %'ld ~ %'ld m per range.\n", M_inc, count_valid_m);
            printf("\tOverflowing to CPU when less than %lu active m\n\n", overflow_count);
            setlocale(LC_NUMERIC, "C");

            //printf("\tStarting to create GPU Batches\n");
        }

        StatsCounters stats(high_resolution_clock::now());
        GpuStatsCounters gpu_stats;
        gpu_stats.s_per_batch = 1; // lie to avoid issue on first pass.

        /* Note: Uses a double batched system
         * C++ Thread is preparing batch_a (even more m), while GPU runs batch_b */
        std::array<GPUBatch, GPU_BATCHES> gpu_batches = {
            GPUBatch(GPU_BATCH_SIZE),
            GPUBatch(GPU_BATCH_SIZE),
            GPUBatch(GPU_BATCH_SIZE),
        };

        std::thread gpu_threads[GPU_BATCHES];
        for(size_t i = 0; i < GPU_BATCHES; i++) {
            gpu_threads[i] = std::thread(run_gpu_thread, og_config.verbose, i, std::ref(gpu_batches[i]));
        }

        // Silly but that's what life is.
        std::queue<int> open_gpu;
        uint64_t running_batches = 0;

        // Main loop
        while (is_running && stop_queue <= 1) {
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
                    usleep(5'000); // 10ms
                    gpu_stats.wait_not_active++;
                    continue;
                }

                if (sieve_data->unknown_m_i.empty()) {
                    if (!sieve_data->try_set_unknowns()) {
                        lock.unlock();
                        usleep(1'000); // 1ms
                        gpu_stats.wait_no_next_tests++;
                        continue;
                    }
                    assert( sieve_data->unknown_m_i.size() );
                    assert( sieve_data->test_i == 0 );
                }

                for (size_t i = 0; i < GPU_BATCHES; i++) {
                    GPUBatch& batch = gpu_batches[i];
                    batch.mutex.lock();

                    if (batch.state != GPUBatch::State::EMPTY) {
                        batch.mutex.unlock();
                        continue;
                    }

                    batch.fill_start = high_resolution_clock::now();
                    fill_batch(i, batch, K, t, sieve_data->current_testing_x);
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
                    printf("HOW DID THIS HAPPEN? No open_gpu?");
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
                    stats.s_total_prp_tests += batch.i;
                    stats.s_total_primes += primes_in_batch;

                    if (og_config.verbose >= 4) {
                        printf("\tGot Finished Batch(%d)=%u prime | %lu running, %lu/%lu\n",
                                i, primes_in_batch,
                                running_batches, d->test_i, d->unknown_m_i.size());
                    }

                    // if all batches finished then move to next set
                    if (running_batches == 0 && d->test_i && d->test_i == d->unknown_m_i.size()) {
                        // TODO technically this is less than cpu_fraction at start of X
                        if (sieve_data->active_m_i.size() < overflow_count) {
                            push_to_overflow_and_increment_M_range(stats);
                            if (stop_queue)
                                stop_queue += 1;
                        } else {
                            sieve_data->increment_X();
                        }
                    }
                    sieve_mtx.unlock();

                    batch.results_end = high_resolution_clock::now();

                    double ms_fill = duration<double>(batch.fill_end - batch.fill_start).count();
                    double ms_queued_full = duration<double>(batch.gpu_start - batch.fill_end).count();
                    double ms_run = duration<double>(batch.gpu_end - batch.gpu_start).count();
                    double ms_queued_done = duration<double>(batch.results_start - batch.gpu_end).count();
                    double ms_results = duration<double>(batch.results_end - batch.results_start).count();
                    double total_t = duration<double>(high_resolution_clock::now() - stats.s_start_t).count();

                    gpu_stats.batches_run += 1;
                    gpu_stats.batches_partial += (batch.i < GPU_BATCH_SIZE);
                    gpu_stats.d_fill += ms_fill;
                    gpu_stats.d_queued_full += ms_queued_full;
                    gpu_stats.d_run += ms_run;
                    gpu_stats.d_queued_done += ms_queued_done;
                    gpu_stats.d_results += ms_results;
                    gpu_stats.s_per_batch = total_t / gpu_stats.batches_run;

                    //if (rand() % (1 * 1024) == 0) {
                    if (0) {
                        // TODO check if gpu times are the same.
                        // If so that means that they are running side by side which maybe isn't what we want.
                        printf("CPU: batch timing fill: %.4f, to gpu: %.4f, "
                                "gpu: %.4f, to cpu: %.4f, processing results: %.4f\n",
                                ms_fill, ms_queued_full, ms_run, ms_queued_done, ms_results);
                    }

                    // Result batch to EMPTY
                    batch.state = GPUBatch::State::EMPTY;
                }
            }
        }

        // ----- cleanup
        {
            mpz_clear(K);
            mpz_clear(t);
        }

        if (og_config.verbose >= 1) {
            double total_t = duration<double>(high_resolution_clock::now() - stats.s_start_t).count();
            setlocale(LC_NUMERIC, "");
            printf("\nGPU Timings:\n");
            printf("\tm processed    : %'lu (%'u/sec)\n",
                    stats.s_tests, (uint32_t) (stats.s_tests / total_t));
            printf("\ttotal tests    : %'lu (%.1f%% prime) (%'u/sec)\n",
                    stats.s_total_prp_tests,
                    100.0 * stats.s_total_primes / stats.s_total_prp_tests,
                    (uint32_t) (stats.s_total_prp_tests / total_t));
            printf("\ttotal batches   : %'lu (%.5f secs/batch)\n",
                    gpu_stats.batches_run, gpu_stats.s_per_batch);
            printf("\twaits on no active_m(5ms) : %ld\n", gpu_stats.wait_not_active);
            printf("\twaits on no next_tests(1ms): %ld\n", gpu_stats.wait_no_next_tests);
            printf("\tfilling batches: %.1f seconds (%.1f%%)\n",
                    gpu_stats.d_fill, 100 * gpu_stats.d_fill / total_t);
            printf("\twaiting filled : %.1f seconds (%.1f%%)\n",
                    gpu_stats.d_queued_full, 100 * gpu_stats.d_queued_full / total_t);
            printf("\trunning on gpu : %.1f seconds (%.1f%%)\n",
                    gpu_stats.d_run, 100 * gpu_stats.d_run / total_t);
            printf("\twaiting done   : %.1f seconds (%.1f%%)\n",
                    gpu_stats.d_queued_done, 100 * gpu_stats.d_queued_done / total_t);
            printf("\tresults        : %.1f seconds (%.1f%%)\n",
                    gpu_stats.d_results, 100 * gpu_stats.d_results / total_t);
            printf("\tbatch fill %%   : %.1f%% (%% fill), %.1f%% (%% partial batch)\n",
                    100.0 * stats.s_total_prp_tests / gpu_stats.batches_run / GPU_BATCH_SIZE ,
                    100.0 * gpu_stats.batches_partial / gpu_stats.batches_run
            );
            printf("\n");
            setlocale(LC_NUMERIC, "C");
        }

        if (og_config.verbose >= 3)
            printf("End of testing thread, Joining batch threads\n");
        // Send notifies (to wake up GPU thread and stop conditional waiting)
        for (auto& gpu_batch : gpu_batches) {
            gpu_batch.cv.notify_all();
        }
        size_t i = 0;
        for (auto & gpu_thread : gpu_threads) {
            gpu_thread.join();
            if (og_config.verbose >= 3)
                cout << "\tbatch gpu thread(" << i << ") joined" << endl;
            i++;
        }
    } catch (const std::exception &e) {
        cout << "ERROR in testing_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


int ctrl_c_count = 0;
void signal_callback_handler(int) {
    ctrl_c_count++;
    if (stop_queue == 0) {
       cout << endl;
       cout << "Caught CTRL+C stopping, winding down work." << endl;
       cout << endl;
       stop_queue = 1;
    } else if (ctrl_c_count == 2) {
       cout << endl;
       cout << "Caught 2nd CTRL+C, press one more time to fast exit." << endl;
    } else {
       cout << endl;
       cout << "Caught 3nd CTRL+C, exit(2) now." << endl;
       cout << endl;
       is_running = false;
       exit(2);
    }
}

void prime_gap_test(struct Config config) {
    // Setup test runner
    printf("\n");
#ifdef GPU_TESTING
    printf("TESTING PRIMES ON GPU\n");
    printf("BITS=%d\n", BITS);
    printf("PRP/BATCH=%ld\n", GPU_BATCH_SIZE);
    printf("THREADS/PRP=%d\n", THREADS_PER_INSTANCE);
    printf("GPU_BATCHES=%lu\n", GPU_BATCHES);
#endif // GPU_TESTING

    assert( GPU_BATCH_SIZE == 1024 || GPU_BATCH_SIZE == 2048 || GPU_BATCH_SIZE == 4096 ||
            GPU_BATCH_SIZE == 8192 || GPU_BATCH_SIZE ==16384 || GPU_BATCH_SIZE ==32768 );

    mpz_t K;
    {
        init_K(config, K);

#ifdef GPU
        // +4 is just is personal safety blanket buffer.
        size_t N_bits = mpz_sizeinbase(K, 2) + log2(config.m_start + 1000ul * config.m_inc) + 4;

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
#endif  // GPU
    }

    is_running = true;
    stop_queue = 0;

    // Setup CTRL+C catcher
    signal(SIGINT, signal_callback_handler);

    sieve_data = std::make_unique<SieveData>(config);

    // This has output that's nicer close to the top.
    std::thread testing_thread(run_testing_thread, config);
    usleep(50'000); // 50ms

    // Setup
    {
        auto s_start_t = high_resolution_clock::now();
        sieve_data->setup_sieve_data(false);
        if (config.verbose >= 2) {
            auto s_stop_t = high_resolution_clock::now();
            printf("\tSetup took %.1f seconds\n",
                   duration<double>(s_stop_t - s_start_t).count());
        }
        if (config.verbose >= 1)
            printf("\n");
    }
    std::thread sieve_thread(run_sieve_thread);

    vector<std::thread> overflow_threads;
    for(size_t i = 0; i < (unsigned)config.cpu_threads; i++) {
        overflow_threads.emplace_back(run_cpu_overflow_thread, i, std::ref(K));
    }

    while (is_running && stop_queue == 0) {
        usleep(100'000); // 100ms
    }

    if (config.verbose >= 3)
        cout << "Joining threads" << endl;

    // Tell other threads to quit
    {
        sieve_thread.join();
        if (config.verbose >= 3)
            cout << "\tsieve joined" << endl;
        testing_thread.join();
        if (config.verbose >= 3)
            cout << "\ttesting joined" << endl;

        overflow_cv.notify_all();  // wake up all overflow thread
        for (auto& thread : overflow_threads) {
            thread.join();
        }
        overflow_threads.clear();
        if (config.verbose >= 3)
            cout << "\tAll CPU overflows joined" << endl;
    }

    mpz_clear(K);
}
