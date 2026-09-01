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

#include "gap_search_gpu.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
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
#include "gap_stats.h"
#include "gpu_testing.h"
#include "overflow.h"


#define GPU_SIEVE
//#define GPU_VERIFY

#define CPU_SIEVE (!defined(GPU_SIEVE) || defined(GPU_VERIFY))

#ifdef GPU_SIEVE
#include "gpu_sieve.h"
#endif // GPU_SIEVE




using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using namespace std::chrono;

//*************************************************************************** //
//**********************************GLOBALS********************************** //



/**
 * Try to have completed this many sieve ahead of the GPU
 * Small extra cost of advance_X filtering for any recent primes
 * Big saving when X has few unknowns
 *      -> some ranges have 1/2 as many for some unknown (to seth) reason
 */
const size_t OPEN_SIEVES = 4;

// GLOBALS PART 1
// more globals in part 2

/** Shared state between threads */
std::atomic<bool> is_running;
std::atomic<uint8_t> stop_queue{0};

// Don't read from sieve_data without holding sieve_mtx
std::mutex sieve_mtx;
std::unique_ptr<SieveData> sieve_data;

//*************************************************************************** //


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

    if (100'000 < config.max_prime && config.max_prime > 1'000'000'000) {
        printf("\tmax_prime(%'ld) should be between 100K and 1B\n", config.max_prime);
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

TestData::TestData(const struct Config config)
        : stats(high_resolution_clock::now()) {
    m_inc = config.m_inc;
    verbose = config.verbose;

    // Need to be able to write to entry at [config.minc >> 6]
    found_prime_m_i.resize((m_inc/2) / 32 + 1, 0);
    full_reset();
}

void TestData::full_reset() {
    std::fill(found_prime_m_i.begin(), found_prime_m_i.end(), 0);
    unknown_m_i.clear();
    test_i = 0;

    state = State::WAITING;
}


SieveData::SieveData(const struct Config config) {
    this->config = config;

    sieves_ready = 0;
    next_sieves.resize(OPEN_SIEVES);

    for (auto prime : get_sieve_primes(config.p)) {
        if (config.d % prime == 0)
            D_primes.push_back(prime);
        else
            K_primes.push_back(prime);
    }
    assert(K_primes.back() == config.p);

    // Has roughly p bits -> find out to 10 merit way more than needed.
    uint32_t stop_x = config.p * 10;
    coprime_X = get_coprime_X(config, stop_x);
}

/**
 * Vector of mi, such that gcd(config.m_start + mi, config.d)
 */
void SieveData::is_coprime_and_valid_m() {
    const uint64_t M_start = config.m_start;
    const uint64_t M_inc = config.m_inc;
    assert(M_inc < std::numeric_limits<uint32_t>::max());

    const uint32_t D = config.d;
    assert( D % 2 == 0 );

    active_m_i.clear();

    // M must be coprime to D, removes all evens.
    assert( M_start % 2 == 0 && M_inc % 2 == 0 );
    is_m_coprime.resize(M_inc / 2);
    std::fill(is_m_coprime.begin(), is_m_coprime.end(), 1);

    for (uint32_t p : D_primes) {
        if (p == 2) continue;

        // mark off any m = m_start + mi that shares factor with d
        uint64_t first = (p - (M_start % p)) % p;
        if ((first & 1) == 0) {
            first += p;
        }
        assert((M_start + first) % p == 0);
        assert((M_start + first) % 2 == 1);

        for (uint64_t mi = first >> 1; mi < is_m_coprime.size(); mi += p) {
            is_m_coprime[mi] = 0;
        }
    }

    // Slower than dynamic bitset, but fast enough
    size_t count = std::count(is_m_coprime.begin(), is_m_coprime.end(), 1);
    active_m_i.reserve(count);

    for (uint32_t i = 0; i < is_m_coprime.size(); i++) {
        if (is_m_coprime[i]) {
            //assert(gcd(M_start + mi, D) == 1);
            active_m_i.push_back(2 * i + 1);
        }
    }
}


/** sieve_mtx must be held while calling */
void SieveData::setup_sieve_data(bool stop) {
    // Verify stuff
    assert(state == SieveData::NEW);

    sieve_x_i = 0;
    current_sieve_x = coprime_X[sieve_x_i];
    testing_x_i = 0;
    current_testing_x = coprime_X[testing_x_i];

    sieves_ready = 0;
    sieves_ready.notify_all();
    for (auto& t : next_sieves) {
        t.first = 0;
        t.second.clear();
    }

    if (stop)
        return;

    active_m_i.clear();
    is_coprime_and_valid_m();

    num_valid = active_m_i.size();
    if (config.verbose + (config.m_start <= 1) >= 3)
        printf("\nSetup, starting at X=%lu with %ld/%ld active_m\n",
                current_sieve_x, num_valid, config.m_inc);

    state = SieveData::FIRST_SIEVE;
}

/** Remove any element of A where bit A is set in B. */
static
void remove_vector(vector<uint32_t> &A, const vector<uint32_t> &B) {
    size_t i = 0;
    for (uint32_t a : A) {
        uint32_t index = a >> 1;
        // ~10% faster to alway set A[i] (branchless)
        A[i] = a;
        // If this wasn't in B, increment i
        uint32_t increment = (B[index >> 5] & (1 << (index & 31))) == 0;
        i += increment;
    }
    A.resize(i);
}

/** sieve_mtx, test_data.lock() must be held while calling */
bool SieveData::try_set_testing_data(TestData &testing) {
    if (this->state == NEW) {
        return false;
    }

    assert( testing.state == TestData::WAITING );
    assert( testing.unknown_m_i.size() == 0 );
    assert( testing.running_batches == 0 );

    // Look for finished sieve to copy over.
    for (uint32_t i = 0; i < OPEN_SIEVES; i++) {
        auto &[s_x, test_m_i] = next_sieves[i];
        if (s_x == current_testing_x) {
            s_x = 0;

            // Set testing data.
            testing.m_start = config.m_start;
            testing.testing_x = current_testing_x;
            testing.unknown_m_i.swap(test_m_i);
            // Removing any primes.
            remove_vector(testing.unknown_m_i, testing.found_prime_m_i);

            assert(testing.unknown_m_i.size());

            testing.state = TestData::ACTIVE;
            testing.state.notify_one();

            test_m_i.clear();

            sieves_ready--;
            sieves_ready.notify_all();
            assert(next_sieves[i].first == 0);
            assert(next_sieves[i].second.empty());
            return true;
        }
    }

    return false;
}

/** sieve_mtx must be held while calling */
void SieveData::increment_X() {

    // asserting next sieve is already done.
    testing_x_i++;
    current_testing_x = coprime_X[testing_x_i];
    if (config.verbose >= 3 && current_testing_x % 300 <= 1) {
        printf("\tMoving to X=%ld\n", current_testing_x);
    }
}

/**
 * Return a^-1 mod p
 * a^-1 * a mod p = 1
 * assumes that gcd(a, p) = 1
 */
static int32_t _invert(int32_t a, int32_t p) {
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


static
void run_sieve_thread(std::atomic<uint8_t> &setup_done) {
    try {
        pthread_setname_np(pthread_self(), "SIEVE_THREAD");
        std::ignore = nice(-2); // Increase priority a bit

        std::unique_lock<std::mutex> lock(sieve_mtx, std::defer_lock);

        auto s_thread_start_t = high_resolution_clock::now();

        // Some prework
        mpz_t K;
        struct Config config = sieve_data->config;
        init_K(config, K);

        uint64_t K_mod_d = mpz_fdiv_ui(K, config.d);
        (void)K_mod_d;
        vector<std::pair<uint32_t, uint32_t>> p_and_neg_inverse_k;
        vector<std::pair<uint32_t, uint32_t>> p_and_neg_inverse_k_small;
        vector<std::pair<uint32_t, uint32_t>> p_and_neg_inverse_k_d;
        p_and_neg_inverse_k.reserve(primesieve::count_primes(3, config.max_prime));
        {
            primesieve::iterator iter;
            uint64_t prime = iter.next_prime();
            assert (prime == 2);  // we skip 2 which is the oddest prime.
            for (prime = iter.next_prime(); prime < config.max_prime; prime = iter.next_prime()) {
                if ((prime <= config.p) && (config.d % prime > 0))
                    continue;

                const uint64_t base_r = mpz_fdiv_ui(K, prime);
                assert( 0 < base_r && base_r < prime );
                const int64_t inv_K = _invert(base_r, prime);
                assert( 0 < inv_K && ((uint32_t) inv_K) < prime );
                assert( (inv_K * base_r) % prime == 1 );
                const uint32_t neg_inv_K = prime - inv_K;

                if ((prime <= config.p) && config.d % prime == 0) {
                    p_and_neg_inverse_k_d.emplace_back(prime, neg_inv_K);
                } else if (prime < 100'000) {
                    p_and_neg_inverse_k_small.emplace_back(prime, neg_inv_K);
                } else {
                    p_and_neg_inverse_k.emplace_back(prime, neg_inv_K);
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

#if CPU_SIEVE
        const auto M_INC_HALF = m_inc / 2;
        // Need to be able to write to composites[m_inc] as sentinel
        vector<uint32_t> composites(M_INC_HALF / 32 + 2, 0);
#endif  // CPU_SIEVE

        uint64_t D = config.d;
        assert( 16 * D % 32 == 0);
        vector<uint32_t> d_wheel(16*D / 32, 0);
        vector<uint32_t> d_indexes;
        uint64_t total_m = 0;
        uint64_t total_runs = 0;
        uint64_t total_active = 0;
        uint64_t total_unknown = 0;
        double avg_primes = 0;
        double total_time = 0;
        double wheel_time = 0;
        double finalize_time = 0;

        uint64_t max_p_i = p_and_neg_inverse_k.size();

        setup_done = 1;
        setup_done.notify_all();

        while (is_running && stop_queue <= 1) {
            lock.lock();
            uint64_t X = sieve_data->current_sieve_x;
            const auto state = sieve_data->state;
            if (state == SieveData::FIRST_SIEVE) {
                total_m += m_inc;
                max_p_i = p_and_neg_inverse_k.size();
                if (config.m_start != sieve_data->config.m_start) {
                    if (config.verbose >= 3)
                        printf("Reset GPU Sieve to 0\n");
                }
            }

            if ((state != SieveData::FIRST_SIEVE && state != SieveData::ACTIVE)
                    || X == 0) {
                lock.unlock();
                usleep(1'000); // 1ms
                continue;
            }

            // Check if any empty next_sieves.
            {
                if (sieve_data->sieves_ready == OPEN_SIEVES) {
                    lock.unlock();
                    sieve_data->sieves_ready.wait(OPEN_SIEVES);
                    continue;
                }

                // lower max_p_i if not many sieves ready
                if (sieve_data->sieves_ready < 2 && sieve_data->sieve_x_i > 20) {
                    max_p_i -= max_p_i / 10;
                    if (max_p_i < 10000)
                        max_p_i = 10000;
                }
            }

            config = sieve_data->config;

            lock.unlock();

            auto s_start_t = high_resolution_clock::now();
            double wheel_duration_t = 0;
            const uint64_t m_start = config.m_start;

            assert (D % 2 == 0);

            // M must be coprime with D  ->  M must be Odd
            // M_start -> even  ->  M_i = odd
            // M(odd) * K(odd) + X  ->  X must be even
            assert (X % 2 == 0);

            assert(m_start % 2 == 0); // or fix the code

            uint64_t prime = 0;

#if CPU_SIEVE
            // Don't need fill because wheel sets (not or's)
            //std::fill(composites.begin(), composites.end(), 0);

            { // Handle all divisors of d at one time.
                auto s_wheel_t = high_resolution_clock::now();
                assert(D < config.m_inc / 100); // Do something different if not true.

                std::fill(d_wheel.begin(), d_wheel.end(), 0);
                uint32_t d_wheel_bits = 32 * d_wheel.size();

                for( const auto& [d, neg_inv_K] : p_and_neg_inverse_k_d) {
                    if (d == 2) continue;

                    // m % d != 0, K % d != 0, if X % d == 0, (m*K + X) % d != 0
                    // Skipping these doesn't save any time and makes verification harder.
                    //if (X % d == 0) continue;

                    // Need m_start % d to not overflow.

                    uint64_t mi_0 = (X * neg_inv_K + d - (m_start % d)) % d;
                    mi_0 += (mi_0 & 1) ? 0 : d;
                    assert( ((m_start + mi_0) * K_mod_d + X) % d == 0 );

                    // mark all later multiples
                    for( uint32_t i = mi_0 >> 1; i < d_wheel_bits; i += d ) {
                        d_wheel[i >> 5] |= 1 << (i & 31);
                    }
                }

                // d_wheel tiled till it's a multiple of 32 bits.
                for(uint32_t c_i = 0; c_i < composites.size(); ) {
                    size_t copy = std::min(composites.size() - c_i, d_wheel.size());
                    for (uint32_t j = 0; j < copy; j ++) {
                        composites[c_i + j] = d_wheel[j];
                    }
                    c_i += copy;
                }
                wheel_duration_t = duration<double>(high_resolution_clock::now() - s_wheel_t).count();
            }

            if (1) {
                // Break the larger range up into smaller ranges that are more likely to fit in L2 (2MB cache)
                uint64_t intervals = M_INC_HALF / 995'000 + 1;
                //printf("Breaking up into %lu intervals of %lu each\n", intervals, m_inc / intervals);
                for (size_t interval = 0; interval < intervals; interval++) {
                    // indexed into [0, m_inc]
                    uint64_t i_start = interval * M_INC_HALF / intervals;
                    uint64_t i_end   = (interval+1) * M_INC_HALF / intervals;
                    // to replace m_start
                    uint64_t m_interval_start = m_start + (i_start << 1);

                    //printf("%2lu -> [%lu, %lu) = %lu\n", interval, i_start, i_end, interval_half_length);
                    for( const auto& [p, neg_inv_K] : p_and_neg_inverse_k_small) {
                        prime = p;
                        // if ((m * K + X) % p == 0) {
                        // if ((m * K) % p == -X) {
                        // if ((m * K * inv_K) % p == -X * inv_K) {
                        //uint64_t m_0 = (-X * inv_K) % prime;

                        uint64_t i_start_shift = prime - (m_interval_start % prime);
                        uint64_t mi_0 = (X * neg_inv_K + i_start_shift) % prime;

                        // This requires K odd and m_start even (both checked above)
                        // See 1ba32111 for more details.
                        mi_0 += (mi_0 & 1) ? 0 : prime;
                        mi_0 >>= 1; // Divide by 2 (even indexes aren't stored)

                        for ( uint32_t t = i_start + mi_0; t < i_end; t += prime) {
                            composites[t >> 5] |= 1 << (t & 31);
                        }
                    }
                }
            } else {
                for( const auto& [p, neg_inv_K] : p_and_neg_inverse_k_small) {
                    prime = p;
                    // if ((m * K + X) % p == 0) {
                    // if ((m * K) % p == -X) {
                    // if ((m * K * inv_K) % p == -X * inv_K) {
                    //uint64_t m_0 = (-X * inv_K) % prime;

                    // Just needs to be small, not exact, 2nd mod makes exact
                    uint64_t m_start_shift = prime - (m_start % prime);
                    uint64_t mi_0 = (X * neg_inv_K + m_start_shift) % prime;

                    // This requires K odd and m_start even (both checked above)
                    // See 1ba32111 for more details.
                    mi_0 += (mi_0 & 1) ? 0 : prime;
                    mi_0 >>= 1; // Divide by 2 (even indexes aren't stored)

                    for ( uint32_t t = mi_0; t < M_INC_HALF; t += prime) {
                        composites[t >> 5] |= 1 << (t & 31);
                    }
                }
            }

            /**
             * Optimizations
             *
             * if primes > M_INC_HALF can change loop to single unconditional set statement
             *      This is probably good but isn't relevant for the wide ranges currently being tested.
             *
             * Tried loop unrolling composites[t >> 5] |= 1 << (t & 31) in _small loop
             *      No effect.
             */
            uint64_t neg_inv_K;
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

                    for (uint32_t t = mi_0; t < M_INC_HALF; t += prime) {
                        composites[t >> 5] |= 1 << (t & 31);
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

                    mi_0 += (mi_0 & 1) ? 0 : prime;
                    mi_0 >>= 1;

                    for (uint32_t t = mi_0; t < M_INC_HALF; t += prime) {
                        composites[t >> 5] |= 1 << (t & 31);
                    }
                }
            }
#endif  // CPU_SIEVE

#ifdef GPU_SIEVE
            // TODO get exact prime counts to agree.
            uint8_t *gpu_composites = gpu_sieve.run(
                    m_start, m_inc, X, max_p_i + p_and_neg_inverse_k_small.size());
            prime = p_and_neg_inverse_k[max_p_i-1].first;
#endif // GPU_SIEVE

#ifdef GPU_VERIFY
            {
                uint32_t num_cpu_composite = 0;
                for (auto c : composites) {
                    num_cpu_composite += __builtin_popcount(c);
                }
                uint32_t num_gpu_composite = 0;
                for (uint32_t m_i = 0; m_i < (M_INC_HALF+7)/8; m_i += 1) {
                    num_gpu_composite += __builtin_popcount(gpu_composites[m_i]);
                }

                uint32_t mismatches = 0;
                for (uint32_t m_i = 1; m_i < m_inc; m_i += 2) {
                    uint32_t t = m_i >> 1;
                    uint8_t cpu_bit = (    composites[t >> 5] & (1 << (t & 31))) > 0;
                    uint8_t gpu_bit = (gpu_composites[t >> 3] & (1 << (t & 7))) > 0;
                    bool mismatch = gpu_bit != cpu_bit;
                    mismatches += mismatch;
                    if (mismatch && mismatches < 10) {
                        printf("Mismatch at m=%lu (%u) | X=%lu | CPU: %u, GPU: %u\n",
                                m_start + m_i, m_i, X, cpu_bit, gpu_bit);
                    }
                }
                printf("GPU/CPU sieve mismatches: %u | composites CPU: %u GPU: %u\n",
                        mismatches, num_cpu_composite, num_gpu_composite);
                if (mismatches) {
                    is_running = false;
                    exit(0);
                }
            }
#endif  // GPU_VERIFY

            auto s_stop_t = high_resolution_clock::now();
            double sieve_duration_t = duration<double>(s_stop_t - s_start_t).count();
            total_runs += 1;
            total_time += sieve_duration_t;

            lock.lock();

            if ( X != sieve_data->current_sieve_x) {
                //printf("At the end of sieve X=%u vs current=%u | state=%u, stop=%u\n",
                //        X, sieve_data->current_sieve_x, state
                assert( sieve_data->state == SieveData::FIRST_SIEVE || stop_queue >= 2);
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
                        sieve_data->sieves_ready++;
                        sieve_data->sieves_ready.notify_all();
                        break;
                    }
                }
                assert( tests != nullptr );
                assert( tests->empty() );

                assert( sieve_data->active_m_i.size() );
                assert( sieve_data->active_m_i.back() < m_inc );

                /**
                 * Various optimization ideas
                 * 1. Avoid push_back by having this be uint32_t *
                 * 2. Uncondiontal set on test[i] (need to resize before hand?)
                 * 3. store m_inc as uint16_t?
                 * 4. Try to reduce initial active_m_i?
                 */
                // "most" (80-90%+) should be composite, so this keep ~14-30%
                // Very similiar to `remove_vector`.
                for (auto m_i : sieve_data->active_m_i) {
                    //assert(m_i < m_inc);
                    //assert(m_i & 1 == 1); // M is odd, M start is even, m_i must be odd.
                    uint32_t t = m_i >> 1;
#ifdef GPU_SIEVE
                    if (!(gpu_composites[t >> 3] & (1 << (t & 7))))
#else
                    if (!(    composites[t >> 5] & (1 << (t & 31))))
#endif  // GPU_SIEVE
                        tests->push_back(m_i);
                }

                unknowns_size = tests->size();
                assert( active_size < 1'000 || unknowns_size > 0 );
                avg_primes += 1.0 * prime * active_size;
                total_active += active_size;
                total_unknown += unknowns_size;

                auto s_stop_t = high_resolution_clock::now();
                finalize_duration_t = duration<double>(s_stop_t - s_start_t).count();
                wheel_time += wheel_duration_t;
                finalize_time += finalize_duration_t;
                total_time += finalize_duration_t;

                // Move to next range.
                sieve_data->sieve_x_i++;
                sieve_data->current_sieve_x = sieve_data->coprime_X[sieve_data->sieve_x_i];
                if (state == SieveData::FIRST_SIEVE) {
                    // Mark as active after next_sieves is set.
                    sieve_data->state = SieveData::ACTIVE;
                }
            }

            if ((config.verbose + (X <= 2) + (config.m_start <= 1'000'000)) >= 3) {
#if CPU_SIEVE
                printf("\tSieve @X=%lu with %u/%u (%.0f%%) unknown/active last prime=%lu"
                       " took %.3f (wheel: %.3f) + %.3f seconds\n",
                       X, unknowns_size, active_size,
                       100.0 * unknowns_size / active_size,
                       prime,
                       sieve_duration_t, wheel_duration_t, finalize_duration_t);
#else
                printf("\tSieve @X=%lu with %u/%u (%.0f%%) unknown/active"
                       " took %.3f + %.3f seconds\n",
                       X, unknowns_size, active_size,
                       100.0 * unknowns_size / active_size,
                       sieve_duration_t, finalize_duration_t);
#endif  // CPU_SIEVE
            }

            lock.unlock();
        }

        mpz_clear(K);

        if (config.verbose >= 1 && total_runs > 0) {
            double total_s = duration<double>(high_resolution_clock::now() - s_thread_start_t).count();
            setlocale(LC_NUMERIC, "");
            printf("\nSIEVE Timings:\n");
            printf("\tsieves : %lu (%.3f/second, %.1f%% of total time)\n",
                    total_runs, total_runs / total_s, 100.0 * total_time / total_s);
            printf("\tavg prime: %'lu\n", (uint64_t) (avg_primes / total_active));
            printf("\ttotal active: %'lu, unknown: %'lu (%.2f%%)\n",
                    total_active, total_unknown, 100.0 * total_unknown / total_active);
            printf("\tactive / run: %'lu, unknown / run: %'lu\n",
                    total_active / total_runs, total_unknown / total_runs);
            printf("\t---------------------------------------\n");
            printf("\ttotal time            : %.1f seconds (%.4f/sieve, %.3f secs/billion)\n",
                    total_time, total_time / total_runs, total_time / total_runs * 1e9 / m_inc);
            printf("\twheel time    (%4.1f%%) : %.1f seconds (%.4f/sieve)\n",
                    100 * wheel_time / total_time, wheel_time, wheel_time / total_runs);
            printf("\tfinalize time (%4.1f%%) : %.1f seconds (%.4f/sieve)\n",
                    100 * finalize_time / total_time, finalize_time, finalize_time / total_runs);
            printf("\n");
            setlocale(LC_NUMERIC, "C");
        }
    } catch (const std::exception &e) {
        cout << "ERROR in run_sieve_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


/** sieve_mtx must be held while calling */
void SieveData::push_to_overflow_and_increment_M_range() {
    // assert( test_data->state == TestData::DONE );

    uint64_t m_start = config.m_start;
    uint64_t m_inc = config.m_inc;
    uint32_t min_X = current_testing_x + 1;
    if (config.verbose >= 2) {
        printf("\n\tMoving to M_start from %ld to %ld Queued %lu for overflow(X >%ld)",
                m_start, m_start + m_inc,
                active_m_i.size(),
                current_testing_x);
        // Not as useful given that most things don't live in this queue
        if (overflow.size)
            printf(" (queue had %u already)", overflow.size.load());
        printf("\n");
    }

    if (1) { // Disable when benchmarking sieve
        overflow.lock();
        for (uint32_t m_i : active_m_i) {
            overflow.queue.emplace_back(m_start + m_i, min_X, Overflow::Type::NEXT_PRIME);
        }
        // Safe because I hold overflow.lock()
        overflow.size = overflow.size + active_m_i.size();
        overflow.unlock();
        overflow.size.notify_one();
    }

    config.m_start += m_inc;
    state = SieveData::NEW;

    setup_sieve_data(stop_queue > 0);
}

static
void run_testing_thread(const struct Config og_config) {
    try {
        pthread_setname_np(pthread_self(), "TESTING_THREAD");
        std::ignore = nice(-2); // Increase priority a bit
        cout << endl;

        mpz_t K;
        init_K(og_config, K);
        const uint64_t D = og_config.d;

        const uint64_t count_valid_m = count_num_m(og_config.m_start, og_config.m_inc, D);
        const uint64_t overflow_count = count_valid_m * og_config.cpu_fraction;

        // Print Header info
        if (og_config.verbose >= 1) {
            setlocale(LC_NUMERIC, "");
            const uint64_t M_inc = og_config.m_inc;
            assert(count_valid_m > 0 && count_valid_m <= M_inc);
            printf("\nTesting ranges of %'ld ~ %'ld m per range.\n", M_inc, count_valid_m);
            printf("\tOverflowing to CPU when less than %lu active m\n\n", overflow_count);
            setlocale(LC_NUMERIC, "C");
        }

        /* Note: Uses a double batched system
         * C++ Thread is preparing batch_a (even more m), while GPU runs batch_b */
        std::deque<GPUBatch> gpu_batches;
        for (uint32_t i = 0; i < GPU_BATCHES; i++) {
            gpu_batches.emplace_back(GPU_BATCH_SIZE);
        }
        TestData test_data{og_config};

        std::thread gpu_threads[GPU_BATCHES];
        for(size_t i = 0; i < GPU_BATCHES; i++) {
            // Barely needs gpu_batches to be owned here.
            gpu_threads[i] = std::thread(run_gpu_thread,
                    i, og_config.verbose,
                    std::ref(test_data),
                    std::ref(gpu_batches[i]),
                    std::ref(K)
            );
        }

        // Main loop
        while (is_running && stop_queue <= 1) {
            /**
             * Try to fill all batches
             * queue on gpu all ready batches
             * wait for result from the 1st batch sent
             */

            const auto state = test_data.state.load();

            if (state == TestData::WAITING) {
                sieve_mtx.lock();
                test_data.lock();

                uint8_t had_ready = sieve_data->sieves_ready;
                bool set = sieve_data->try_set_testing_data(test_data);
                if (!set) test_data.gpu_stats.wait_not_active++;

                test_data.unlock();
                sieve_mtx.unlock();

                if (set) {
                    // test_data isn't locked so do this write first before waking up batches.
                    assert( had_ready > 0 );
                    assert( test_data.active_batches == 0 );
                    test_data.active_batches = GPU_BATCHES;
                    // Mark gpu batches as active
                    for (auto& batch : gpu_batches) {
                        assert( batch.state == GPUBatch::WAITING );
                        batch.state = GPUBatch::EMPTY;
                        batch.state.notify_one();
                    }
                } else {
                    auto t0 = high_resolution_clock::now();
                    assert( had_ready == 0 );
                    sieve_data->sieves_ready.wait(0);
                    double wait = duration<double>(high_resolution_clock::now() - t0).count();
                    if (og_config.verbose >= 3) {
                        printf("Wait for sieve: X=%u %.5f seconds\n", test_data.testing_x, wait);
                    }
                    test_data.lock();
                    test_data.gpu_stats.d_wait_not_active += wait;
                    test_data.unlock();
                }

                continue;
            }

            // run_gpu_thread running batches till test_data is done

            assert( state == TestData::ACTIVE || state == TestData::DONE );
            if (state == TestData::ACTIVE ) {
                test_data.wait_for_state_and_lock(TestData::DONE);
            } else {
                test_data.lock();
            }

            if (!is_running) {
                test_data.unlock();
                break;
            }

            {
                // Holding test_data.lock()
                assert( test_data.state == TestData::DONE );
                assert( test_data.running_batches == 0 );
                assert( test_data.test_i == test_data.unknown_m_i.size() );

                // Merge all stats from gpu_stats
                for (auto& batch : gpu_batches) {
                    assert( batch.state == GPUBatch::WAITING );
                    batch.lock();
                    test_data.gpu_stats.merge(batch.stats);
                    batch.stats.reset();
                    batch.unlock();
                }

                sieve_mtx.lock();

                // TODO time this.
                remove_vector(sieve_data->active_m_i, test_data.found_prime_m_i);
                if (sieve_data->active_m_i.size() < overflow_count) {
                    test_data.stats.s_gap_out_of_sieve_next += sieve_data->active_m_i.size();

                    // Mark m_inc tests as having been done, maybe print stats
                    test_data.stats.batches += 1;
                    test_data.stats.total_m += og_config.m_inc;
                    test_data.stats.tested_m += sieve_data->num_valid;
                    if (og_config.verbose >= 1) {
                        test_data.maybe_print_stats();
                    }

                    sieve_data->push_to_overflow_and_increment_M_range();
                    if (stop_queue) {
                        stop_queue++;
                        if (og_config.verbose >= 2) {
                            printf("\tChanged stop_queue to %u\n", stop_queue.load());
                        }
                    }

                    test_data.full_reset();

                } else {
                    sieve_data->increment_X();
                }

                test_data.test_i = 0;
                test_data.unknown_m_i.clear();
                test_data.state = TestData::WAITING;


                sieve_mtx.unlock();
                test_data.unlock();
            }
        }

        // ----- cleanup
        {
            mpz_clear(K);
        }

        if (!is_running) {
            for (auto& batch : gpu_batches) {
                batch.state = GPUBatch::EMPTY;
                batch.state.notify_one();
            }
        }

        // Push note to overflow that we're done.
        for (int32_t i = 0; i < og_config.cpu_threads; i++) {
            overflow.push_to_queue(0, 0, Overflow::Type::STOP_WORKER);
        }

        if (og_config.verbose >= 1) {
            test_data.lock();
            test_data.print_stats();
            test_data.unlock();
        }

        if (og_config.verbose >= 2)
            printf("End of testing thread, Joining batch threads\n");
        // Send notifies (to wake up GPU thread and stop conditional waiting)
        for (auto& gpu_batch : gpu_batches) {
            // Should wait anyone waiting up
            gpu_batch.state = GPUBatch::EMPTY;
            gpu_batch.state.notify_all();
        }
        if (og_config.verbose >= 2)
            cout << "\tjoining gpu threads" << endl;
        size_t i = 0;
        for (auto & gpu_thread : gpu_threads) {
            gpu_thread.join();
            if (og_config.verbose >= 2)
                cout << "\tbatch gpu thread(" << i << ") joined" << endl;
            i++;
        }
        if (og_config.verbose >= 2)
            cout << "\ttesting thread done" << endl;
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
       cout << "Caught 2nd CTRL+C, is_running = false" << endl;
       is_running = false;
    } else {
       cout << endl;
       cout << "Caught 3nd CTRL+C, exit(2) now." << endl;
       exit(2);
       exit(2);
    }
}

void prime_gap_test(struct Config config) {
    // Setup test runner
    printf("\n");

    // Turn into static assert
    assert( GPU_BATCH_SIZE == 1024 || GPU_BATCH_SIZE == 2048 || GPU_BATCH_SIZE == 4096 ||
            GPU_BATCH_SIZE == 8192 || GPU_BATCH_SIZE ==16384 || GPU_BATCH_SIZE ==32768 );

    mpz_t K;
    init_K(config, K);

    gpu_state_and_checks(K, config.m_start + 1000 * config.m_inc);

    is_running = true;
    stop_queue = 0;

    // Setup CTRL+C catcher
    signal(SIGINT, signal_callback_handler);

    sieve_data = std::make_unique<SieveData>(config);

    // Setup
    {
        sieve_mtx.lock();

        auto s_start_t = high_resolution_clock::now();
        sieve_data->setup_sieve_data(false);
        if (config.verbose >= 2) {
            auto s_stop_t = high_resolution_clock::now();
            printf("\tSetup took %.1f seconds\n",
                   duration<double>(s_stop_t - s_start_t).count());
        }
        if (config.verbose >= 1)
            printf("\n");

        sieve_mtx.unlock();
    }
    std::atomic<uint8_t> setup_done{0};
    std::thread sieve_thread(run_sieve_thread, std::ref(setup_done));
    // May take a few seconds for GPUSieve to build up prime lists
    setup_done.wait(0);
    if (config.verbose >= 3)
        printf("Setup Done!\n");

    // This has output that's nicer close to the top.
    std::thread overflow_thread{run_overflow_coordinator_thread, std::ref(config)};
    usleep(10'000); // 50ms

    std::thread testing_thread{run_testing_thread, config};

    while (is_running && stop_queue <= 1) {
        usleep(50'000); // 50ms
    }

    sieve_data->sieves_ready = OPEN_SIEVES / 2;
    sieve_data->sieves_ready.notify_all();

    {
        if (config.verbose >= 2)
            cout << "Joining threads" << endl;

        sieve_thread.join();
        if (config.verbose >= 2)
            cout << "\tsieve joined" << endl;

        testing_thread.join();
        if (config.verbose >= 2)
            cout << "\ttesting joined" << endl;

        overflow_thread.join();
        if (config.verbose >= 2)
            cout << "\toverflow joined" << endl;
    }

    if (config.verbose >= 1 && is_running) {
        printf("\tresume at --mstart=%lu\n", sieve_data->config.m_start);
    }

    mpz_clear(K);
}
