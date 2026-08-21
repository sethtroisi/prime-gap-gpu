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

#include "overflow.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unistd.h>
#include <utility>


// pthread_setname_np
#include <pthread.h>

#include <gmp.h>
#include <primesieve.hpp>

#include "gap_common.h"
#include "gap_stats.h"
#include "gap_search_gpu.h"


using std::cout;
using std::cerr;
using std::endl;
using namespace std::chrono;


bool overflow_should_run() {
    return !is_running || stop_queue >= 2 || overflowed.size();
}

/**
 * [X] do sieve here. (+15-20%)
 * Move that to some struct
 * Create OverflowBatch
 * Push OverflowBatch occasionally to a GPUBatch
 * Access to runner.run_test in gpu_testing
 * Handle prev prime the same way or ???
 */

/** Globals for this class */
vector<uint16_t> coprime_X;
//vector<uint32_t> d_primes;
vector<std::pair<uint32_t, uint32_t>> p_and_neg_r_small;
vector<std::pair<uint32_t, uint32_t>> p_and_neg_r;


void setup_overflow(const struct Config config) {
    // Set to large enough that we don't have to deal with edge-case where it isn't enough
    uint32_t stop_x = 42 * config.p;
    {
        auto X = get_coprime_X(config, stop_x);
        coprime_X.reserve(X.size());
        for (const auto x : X) {
            coprime_X.push_back(x);
        }
    }

    mpz_t K;
    init_K(config, K);
    {
        primesieve::iterator iter;
        uint64_t prime = iter.next_prime();
        assert (prime == 2);  // we skip 2 which is the oddest prime.
        for (prime = iter.next_prime(); prime < 5'000'000; prime = iter.next_prime()) {
            if (prime <= config.p && (config.d % prime != 0))
                continue;

            const uint32_t base_r = mpz_fdiv_ui(K, prime);
            assert( 0 < base_r && base_r < prime );
            const uint32_t neg_base_r = prime - base_r;
            if (prime < stop_x) {
                p_and_neg_r_small.emplace_back((uint32_t) prime, neg_base_r);
            } else {
                p_and_neg_r.emplace_back((uint32_t) prime, neg_base_r);
            }
        }
    }


}


/**
 * [sieve_start, sieve_start + sieve_length)
 */
void sieve_interval_cpu(const uint64_t m,
        const int32_t sieve_start,
        const int32_t sieve_length,
        vector<uint8_t> &composite,
        const uint32_t start_coprime_x_i,
        vector<uint16_t> &out_X) {

    uint16_t bytes = (sieve_length + 7) / 8 + 1;
    // Includes evens for now
    // TODO: test as just index 6KB vs <1KB
    composite.resize(bytes, 0);
    std::fill(composite.begin(), composite.end(), 0);

    const uint8_t SENTINAL_I = sieve_length;

    for (const auto& [p, neg_r] : p_and_neg_r_small) {
        // -(m * K + sieve_start) % r
        int64_t center_mod = ((int64_t) m * neg_r - sieve_start) % p;
        center_mod += (center_mod < 0) ? 0 : p;

        // Could do by 2*p but not worth the extra code IMO
        for (int32_t i = center_mod; i < sieve_length; i += p) {
            composite[i >> 3] |= 1 << (i & 7);
        }
    }

    for (const auto& [p, neg_r] : p_and_neg_r) {
        // -(m * K + sieve_start) % r
        int64_t center_mod = ((int64_t) m * neg_r - sieve_start) % p;
        center_mod += (center_mod < 0) ? p : 0;

        uint32_t i = center_mod < sieve_length ? center_mod : SENTINAL_I;
        composite[i >> 3] |= 1 << (i & 7);
    }

    out_X.clear();
    uint32_t N = coprime_X.size();
    for (uint32_t x_i = start_coprime_x_i; x_i < N; x_i++) {
        uint16_t x = coprime_X[x_i];
        uint16_t j = x - sieve_start;
        if ((composite[j >> 3] & (1 << (j & 7))) == 0) {
            out_X.push_back(x);
        }
    }
}

uint32_t next_prime_distance(
        const uint64_t m, const uint64_t min_x,
        const mpz_t &K, mpz_t &center, mpz_t &tmp,
        vector<uint8_t> &composite_tmp,
        vector<uint16_t> &out_X,
        uint32_t expect) {

    // TODO something better than this
    uint32_t min_x_i = std::distance(
            coprime_X.begin(),
            std::upper_bound(coprime_X.begin(), coprime_X.end(), min_x - 1));

    uint32_t next_possible_x = coprime_X[min_x_i];

    assert( 1 <= min_x_i && min_x_i < coprime_X.size() );
    assert( min_x <= next_possible_x );
    assert( coprime_X[min_x_i-1] < min_x );

    sieve_interval_cpu(
        m, next_possible_x, coprime_X.back() - next_possible_x + 1,
        composite_tmp, min_x_i, out_X);

    assert( out_X.size() > 0 );
    assert( out_X.front() >= next_possible_x );

    mpz_mul_ui(center, K, m);
    for (auto x : out_X) {
        mpz_add_ui(tmp, center, x);
        if (mpz_probab_prime_p(tmp, 20)) {
            assert(x == expect);
            return x;
        }
    }

    printf("\n\nERROR: Didn't find next prime for %lu * K, after %lu tests\n",
            m, out_X.size());


    // Guarentees a mismatch statement after.
    return 100000;
}


void run_cpu_overflow_thread(uint32_t i, const struct Config og_config,
                             const mpz_t &K_in, TestingStats &stats) {
    try {
        {
            char name[16];
            sprintf(name, "CPU_OVERFLOW_%u\n", i);
            pthread_setname_np(pthread_self(), name);
            std::ignore = nice(+10); // Lower priority a bit
        }

        // Pre-allocated
        mpz_t K, center, next_p, prev_p;
        mpz_init_set(K, K_in);
        mpz_init(center);
        mpz_init(next_p);
        mpz_init(prev_p);
        vector<uint8_t> composite_tmp;
        vector<uint16_t> unknown_X;


        double K_log = calc_log_K(og_config);
        const float min_merit = og_config.min_merit;
        const uint32_t P = og_config.p;
        const uint32_t D = og_config.d;

        // See THEORY.md! Added const is small preference for doing less prev_p.
        const float MIN_MERIT_TO_CONTINUE = 2.6 + std::log2(min_merit * std::log(2) + 1);
        const float MIN_GAP_TO_CONTINUE =  MIN_MERIT_TO_CONTINUE * (K_log + log(og_config.m_inc));

        // 2-5x what comes in per batch
        const uint64_t overflow_too_much = og_config.m_inc * og_config.cpu_fraction;


        std::ofstream record_stream(std::format("records_{}.txt", P), std::ios_base::app);

        std::unique_lock<std::mutex> lock(overflow_mtx);
        while (is_running && (stop_queue < 2 || overflowed.size() > 0)) {
            assert(lock.owns_lock());
            // Lock IS NOT held while waiting.
            overflow_cv.wait(lock, overflow_should_run);

            while (is_running && (overflowed.size() || spot_check.size())) {
                assert(lock.owns_lock());

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
                    auto s_start_t = high_resolution_clock::now();
                    if (!mpz_probab_prime_p(next_p, 20)) {
                        printf("\n\n");
                        printf("%lu'th SPOT CHECK FAILED!\n", stats.spot_checked.load());
                        printf("%lu * %u# / %u + %u is not prime!\n",
                                m_and_x.first, P, D, m_and_x.second);
                        printf("\n\n");
                        exit(1);
                    }
                    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
                    stats.d_spot_check += total_s;
                    continue;
                }

                auto m_and_x = overflowed.front(); overflowed.pop_front();
                lock.unlock();

                // Do this before testing to prevent multiple CPU Sieve Queue prints on same tested value.
                stats.tested++;
                auto m = m_and_x.first;
                auto min_x = m_and_x.second;

                auto s_start_t = high_resolution_clock::now();
                uint64_t next_gap = 0;
                if (1) {
                    mpz_mul_ui(center, K, m);
                    mpz_add_ui(next_p, center, min_x);
                    mpz_nextprime(next_p, next_p);
                    mpz_sub(next_p, next_p, center);
                    next_gap = mpz_get_ui(next_p);
                } else {
                    next_gap = next_prime_distance(
                            m, min_x,
                            K, center, next_p,
                            composite_tmp, unknown_X, next_gap);
                }
                double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
                stats.d_next_prime += total_s;

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
                                    test_gap, gap, m, P, D, prev_gap);
                            merit = test_gap / (K_log + log(m));
                        }

                        if (merit > min_merit) {
                            std::string record = std::format(
                                    "{} {:.3f} {} * {}# / {} - {}",
                                    gap, merit, m, P, D, prev_gap);
                            cout << record << endl;

                            lock.lock();
                            record_stream << record << endl;
                            lock.unlock();
                        }
                    }
                }

                lock.lock();
            }
        }

        if (i == 0 && og_config.verbose >= 1) {
            printf("\nCPU OVERFLOW Timing:\n");
            printf("\ttotal tested   : %lu (%.5f/next_prime)\n",
                    stats.tested.load(), stats.d_next_prime / (0.01 + stats.tested));
            printf("\tspot checked   : %lu (%.6f/prob_prime)\n",
                    stats.spot_checked.load(), stats.d_spot_check / (0.01 + stats.spot_checked));
            printf("\tnext prime only: %lu, both sides: %lu\n",
                    stats.skipped_prev.load(), stats.tested_prev.load());
            uint32_t large = stats.greater_than_min_merit;
            if (large) {
                printf("\t> %.1f merit: %u (%lu = %.1f%% bad next_prime)\n",
                        min_merit, large, stats.mismatches.load(),
                        100.0 * stats.mismatches.load() / large);
            }
            printf("\n");
        }

        if (og_config.verbose >= 3) {
            usleep(i * 10'000); // i * 10ms
            printf("\tCPU overflow(%u) done\n", i);
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
