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
#include "gpu_testing.h"


using std::cout;
using std::cerr;
using std::endl;
using namespace std::chrono;


bool overflow_should_run() {
    return !is_running || stop_queue >= 2 || overflowed.size();
}

/**
 * [X] do sieve here. (+15-20%)
 * Store test_x_i, composite in struct
 *     Consider if all overflow should happen at a fixed x_i (maype set in the first sieve)
 * Create OverflowBatch with 4096 structs
 * New FillBatch -> GPUBatch
 * Access to runner.run_test in gpu_testing
 * Handle prev prime the same way or ???
 */

/** Globals for this class */

vector<uint16_t> coprime_X;
// coprime_X[coprime_lookup[t]] >= t
vector<uint16_t> next_coprime_index;
vector<std::pair<uint32_t, uint32_t>> p_and_neg_r_small;
vector<std::pair<uint32_t, uint32_t>> p_and_neg_r;

OverflowBatch overflow_batch;


void setup_overflow(const struct Config config) {
    // TODO TUNE THIS
    // 15 * p, overflows less than .1% of the time.
    uint32_t stop_x = 25 * config.p;
    {
        auto X = get_coprime_X(config, stop_x);
        coprime_X.reserve(X.size());
        for (const auto x : X) {
            coprime_X.push_back(x);
        }

        next_coprime_index.resize(coprime_X.back() + 1, 0xFFFF);
        uint16_t i = 0;
        for (uint16_t x_i = 0; x_i < coprime_X.size(); x_i++) {
            for ( ; i <= coprime_X[x_i]; i++ ) {
                next_coprime_index[i] = x_i;
            }
        }
        for (i = 0; i < next_coprime_index.size(); i++) {
            assert( coprime_X[next_coprime_index[i]] >= i );
        }
    }


    // TODO d wheel to speed up sieve_interval_cpu.

    mpz_t K;
    init_K(config, K);
    {
        primesieve::iterator iter;
        uint64_t prime = iter.next_prime();
        assert (prime == 2);  // we skip 2 which is the oddest prime.
        for (prime = iter.next_prime(); prime < 200'000; prime = iter.next_prime()) {
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

std::mutex file_mutex;

static
void process_result(
        const float min_merit,
        const double K_log, const uint32_t P, const uint32_t D,
        const mpz_t &K, mpz_t &center,

        const uint64_t m,
        double merit, uint64_t gap, uint64_t prev_gap,
        mpz_t &next_p, mpz_t &prev_p,
        mpz_t &tmp, mpz_t &tmp2,
        TestingStats &stats,
        std::ofstream &record_stream) {

    if (merit > min_merit) {
        stats.greater_than_min_merit++;
        // Double check, we only performed a single round of rabin miller on many numbers.
        mpz_mul_ui(center, K, m);
        mpz_sub_ui(prev_p, center, prev_gap);
        mpz_nextprime(next_p, prev_p);
        mpz_sub(next_p, next_p, prev_p);
        uint64_t test_gap = mpz_get_ui(next_p);
        if (test_gap != gap) {
            // These numbers are marked "prime" by GPU because we only do 1 round.
            mpz_sub_ui(tmp, next_p, 1);
            mpz_set_ui(tmp2, 2);
            // Check if mismatch is Fermat pseudoprime base 2 <=> 2^(np-1) % np = 1
            mpz_powm(tmp, tmp2, tmp, next_p);
            if ( mpz_cmp_ui(tmp, 1) == 0) {
                stats.pseudoprimes++;
                printf("\tFermat Pseuodprime: %lu * %u# / %u + %lu\n",
                        m, P, D, test_gap - prev_gap);
            } else {
                stats.mismatches++;
                printf("\tGAP MISMATCH! %lu vs %lu at %lu * %u# / %u + %lu\n",
                        test_gap, gap, m, P, D, test_gap - prev_gap);
            }
            merit = test_gap / (K_log + log(m));
        }

        if (merit > min_merit) {
            std::string record = std::format(
                    "{} {:.3f} {} * {}# / {} - {}",
                    gap, merit, m, P, D, prev_gap);
            cout << record << endl;

            file_mutex.lock();
            record_stream << record << endl;
            file_mutex.unlock();
        }
    }
}


/** Expects center to be correctly set */
static
void handle_next_prime_result(
        const float MIN_GAP_TO_CONTINUE,
        const float min_merit,
        const double K_log, const uint32_t P, const uint32_t D,
        const mpz_t &K, mpz_t &center,

        const uint64_t m, const uint32_t next_gap,
        mpz_t &next_p, mpz_t &prev_p,
        mpz_t &tmp, mpz_t &tmp2,
        TestingStats &stats,
        std::ofstream &record_stream) {

    if (next_gap < MIN_GAP_TO_CONTINUE) {
        stats.skipped_prev++;
        return;
    }

    stats.tested_prev++;
    mpz_prevprime(prev_p, center);
    mpz_sub(prev_p, center, prev_p);
    uint64_t prev_gap = mpz_get_ui(prev_p);
    uint64_t gap = prev_gap + next_gap;
    double merit = gap / (K_log + log(m));

    process_result(
        min_merit, K_log, P, D, K, center,
        m, merit, gap, prev_gap,
        next_p, prev_p, tmp, tmp2, stats,
        record_stream);
}

/**
 * [sieve_start, sieve_start + sieve_length)
 */
static
void sieve_interval_cpu(const uint64_t m,
        const uint32_t sieve_start,
        const uint32_t sieve_length,
        vector<uint8_t> &composite) {

    uint16_t bytes = (sieve_length + 7) / 8 + 1;
    composite.resize(bytes, 0);
    std::fill(composite.begin(), composite.end(), 0);

    // only interested in even i
    assert(sieve_start % 2 == 0);
    assert(m > p_and_neg_r.back().first);

    for (const auto& [p, neg_r] : p_and_neg_r_small) {
        // -(m * K + sieve_start) % r
        uint64_t center_mod = ((uint64_t) m * neg_r - sieve_start) % p;
        center_mod += (center_mod & 1) ? p : 0;

        uint32_t two_p = p << 1;
        for (uint32_t i = center_mod; i < sieve_length; i += two_p) {
            composite[i >> 3] |= 1 << (i & 7);
        }
    }

    for (const auto& [p, neg_r] : p_and_neg_r) {
        // -(m * K + sieve_start) % r
        uint64_t center_mod = ((uint64_t) m * neg_r - sieve_start) % p;
        if (center_mod < sieve_length && (center_mod & 1) == 0) {
            composite[center_mod >> 3] |= 1 << (center_mod & 7);
        }
    }
}

static
uint32_t next_prime_distance(
        const uint64_t m, const uint64_t min_x,
        const mpz_t &K, mpz_t &center, mpz_t &tmp,
        vector<uint8_t> &composite_tmp,
        TestingStats &stats) {
    auto s_start_t = high_resolution_clock::now();

    uint32_t min_x_i = next_coprime_index[min_x];
    uint32_t next_possible_x = coprime_X[min_x_i];

    assert( 1 <= min_x_i && min_x_i < coprime_X.size() );
    assert( min_x <= next_possible_x );
    assert( coprime_X[min_x_i-1] < min_x );

    sieve_interval_cpu(
        m, next_possible_x, coprime_X.back() - next_possible_x + 1,
        composite_tmp);

    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
    stats.d_next_prime_sieve += total_s;

    mpz_mul_ui(center, K, m);

    const uint32_t N = coprime_X.size();
    for (uint32_t x_i = min_x_i; x_i < N; x_i++) {
        uint16_t x = coprime_X[x_i];
        uint16_t j = x - next_possible_x;
        if ((composite_tmp[j >> 3] & (1 << (j & 7))) == 0) {
            mpz_add_ui(tmp, center, x);
            if (mpz_probab_prime_p(tmp, 20)) {
                return x;
            }
        }
    }

    // Fallback to mpz_nextprime if very large
    mpz_nextprime(tmp, tmp);
    mpz_sub(tmp, tmp, center);
    return mpz_get_ui(tmp);
}

static
void push_to_overflow_batch(
        const uint64_t m, const uint64_t min_x,
        const mpz_t &K, mpz_t &center, mpz_t &tmp,
        vector<uint8_t> &composite_tmp,
        TestingStats &stats) {

    auto s_start_t = high_resolution_clock::now();

    uint32_t min_x_i = next_coprime_index[min_x];
    uint32_t sieve_start = coprime_X[min_x_i];
    assert( min_x <= sieve_start);

    GPUBatch &gpu_batch = overflow_batch.gpu_batch;

    uint32_t i = overflow_batch.i; // start search here.
    for (; i < overflow_batch.N; i++) {
        if (gpu_batch.active[i] == 0) {
            break;
        }
    }

    overflow_batch.i = i+1;
    overflow_batch.added++;
    assert(i < overflow_batch.N);
    assert(gpu_batch.active[i] == 0);

    sieve_interval_cpu(
        m, sieve_start, coprime_X.back() - sieve_start + 1,
        overflow_batch.composite_tmp[i]);

    overflow_batch.data[i] = std::make_tuple(m, min_x_i, sieve_start);
    mpz_mul_ui(center, K, m);
    mpz_add_ui(*overflow_batch.gpu_batch.z[i], center, sieve_start);
    overflow_batch.gpu_batch.active[i] = true;

    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
    stats.d_next_prime_sieve += total_s;
}

static
uint32_t run_overflow_batch(
        const float MIN_GAP_TO_CONTINUE,
        const float min_merit,
        const double K_log, const uint32_t P, const uint32_t D,
        const mpz_t &K, mpz_t &center,

        mpz_t &next_p, mpz_t &prev_p,
        mpz_t &tmp, mpz_t &tmp2,
        TestingStats &stats,
        std::ofstream &record_stream) {

    GPUBatch &gpu_batch = overflow_batch.gpu_batch;
    for (uint32_t i = 0; i < overflow_batch.N; i++) {
        // May need to disable for last batch
        assert( gpu_batch.active[i] == 1 );
    }

    gpu_batch.i = overflow_batch.added;
    std::fill(gpu_batch.result.begin(), gpu_batch.result.end(), -1);

    // Run gpu_batch on GPU.
    auto s_start_t = high_resolution_clock::now();
    one_shot_batch( gpu_batch );
    stats.d_next_prime_gpu += duration<double>(high_resolution_clock::now() - s_start_t).count();

    uint32_t found_primes = 0;
    for (uint32_t i = 0; i < overflow_batch.N; i++) {
        if (!gpu_batch.active[i])
            continue;

        assert (gpu_batch.result[i] == 0 || gpu_batch.result[i] == 1);
        auto& [m, x_i, sieve_start] = overflow_batch.data[i];

        if (gpu_batch.result[i] == 1) {
            // Found nextprime for m!
            overflow_batch.remove_entry(i);
            found_primes++;
            stats.tested_gpu++;

            mpz_mul_ui(center, K, m);
            uint32_t next_gap = coprime_X[x_i];
            if (0) {
                // Debug these results
                mpz_nextprime(tmp, center);
                if ( mpz_cmp(tmp, *gpu_batch.z[i]) != 0 ) {
                    mpz_sub(tmp, tmp, center);
                    uint32_t t = mpz_get_ui(tmp);
                    printf("Disagreement on next_prime for m=%lu | %u vs %u\n",
                            m, next_gap, t);
                }
            }

            handle_next_prime_result(
                MIN_GAP_TO_CONTINUE, min_merit, K_log, P, D, K, center,
                m, next_gap,
                next_p, prev_p, tmp, tmp2, stats, record_stream);

        } else {
            // Advance to next test see `next_prime_distance`
            uint32_t last_x = coprime_X[x_i];
            uint32_t M = coprime_X.size();
            x_i++;
            for (; x_i < M; x_i++) {
                uint16_t x = coprime_X[x_i];
                uint16_t j = x - sieve_start;
                if ((overflow_batch.composite_tmp[i][j >> 3] & (1 << (j & 7))) == 0) {
                    mpz_add_ui(*gpu_batch.z[i], *gpu_batch.z[i], x - last_x);
                    break;
                }
            }
            if (x_i >= M) {
                overflow_batch.remove_entry(i);
                found_primes++;

                // Do manually here
                auto s_start_t = high_resolution_clock::now();
                mpz_sub_ui(center, *gpu_batch.z[i], last_x);
                mpz_nextprime(tmp, *gpu_batch.z[i]);
                mpz_sub(tmp, tmp, center);
                uint32_t next_gap = mpz_get_ui(tmp);
                assert( next_gap > coprime_X.back() );
                stats.d_next_prime_cpu += duration<double>(high_resolution_clock::now() - s_start_t).count();
                stats.tested_cpu++;

                handle_next_prime_result(
                    MIN_GAP_TO_CONTINUE, min_merit, K_log, P, D, K, center,
                    m, next_gap,
                    next_p, prev_p, tmp, tmp2, stats, record_stream);
            }
        }
    }

    //printf("\tRan overflow on GPU found: %lu primes\n", overflow_batch.N - overflow_batch.added);
    return found_primes;
}


// gap, prev_gap, merit
static
void run_tests_on_cpu(
        const float MIN_GAP_TO_CONTINUE, const float min_merit,
        const double K_log, const uint32_t P, const uint32_t D,
        const uint64_t m, const uint64_t min_x,
        const mpz_t &K, mpz_t &center,
        mpz_t &next_p, mpz_t &prev_p,
        mpz_t &tmp, mpz_t &tmp2,
        vector<uint8_t> &composite_tmp,
        TestingStats &stats,
        std::ofstream &record_stream) {

    auto s_start_t = high_resolution_clock::now();
    uint64_t next_gap = 0;
    if (0) {
        mpz_mul_ui(center, K, m);
        mpz_add_ui(next_p, center, min_x);
        mpz_nextprime(next_p, next_p);
        mpz_sub(next_p, next_p, center);
        next_gap = mpz_get_ui(next_p);
    } else {
        next_gap = next_prime_distance(
                m, min_x,
                K, center, tmp,
                composite_tmp, stats);
    }
    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
    stats.d_next_prime_cpu += total_s;
    stats.tested_cpu += 1;

    handle_next_prime_result(
        MIN_GAP_TO_CONTINUE, min_merit, K_log, P, D,
        K, center,
        m, next_gap,
        next_p, prev_p, tmp, tmp2, stats, record_stream);
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

        //assert(i == 0); // With current GPU setup.

        // Pre-allocated
        mpz_t K, center, next_p, prev_p, tmp, tmp2;
        mpz_init_set(K, K_in);
        mpz_inits(center, next_p, prev_p, tmp, tmp2, NULL);
        vector<uint8_t> composite_tmp;


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

                if (1) {
                    run_tests_on_cpu(
                            MIN_GAP_TO_CONTINUE, min_merit, K_log, P, D,
                            m, min_x, K,
                            center, next_p, prev_p,
                            tmp, tmp2, composite_tmp, stats, record_stream);
                } else {
                    lock.lock();
                    push_to_overflow_batch(
                        m, min_x,
                        K, center, tmp,
                        composite_tmp, stats);

                    if (overflow_batch.added == overflow_batch.N) {
                        for (size_t i = 0; i < 10; i++) {
                            auto found = run_overflow_batch(
                                    MIN_GAP_TO_CONTINUE,
                                    min_merit, K_log, P, D, K, center,
                                    next_p, prev_p, tmp, tmp2, stats,
                                    record_stream);
                            if (found) break;
                            printf("GPUBatch didn't find any primes?\n");
                        }
                    }

                    lock.unlock();
                }

                lock.lock();
            }
        }


        if (is_running && overflow_batch.added) { // Clear out any remaining items queued in GPUBatch
            assert(lock.owns_lock());
            if (og_config.verbose >= 1) {
                printf("\tCPU OVERFLOW Finishing %lu remaining items in GPUBatch\n", overflow_batch.added);
            }

            // Slightly akward to run partial batches so handle on CPU.
            GPUBatch &gpu_batch = overflow_batch.gpu_batch;
            for (uint32_t i = 0; i < overflow_batch.N; i++) {
                if (!gpu_batch.active[i])
                    continue;

                auto& [m, x_i, sieve_start] = overflow_batch.data[i];
                { // Handle one row from GPUBatch manually.
                    auto s_start_t = high_resolution_clock::now();
                    uint32_t last_x = coprime_X[x_i];
                    mpz_sub_ui(center, *gpu_batch.z[i], last_x);
                    mpz_nextprime(tmp, *gpu_batch.z[i]);
                    mpz_sub(tmp, tmp, center);
                    uint32_t next_gap = mpz_get_ui(tmp);
                    stats.d_next_prime_cpu += duration<double>(high_resolution_clock::now() - s_start_t).count();
                    stats.tested_cpu++;

                    handle_next_prime_result(
                        MIN_GAP_TO_CONTINUE, min_merit, K_log, P, D, K, center,
                        m, next_gap,
                        next_p, prev_p, tmp, tmp2, stats, record_stream);
                }
                overflow_batch.remove_entry(i);
            }
            lock.unlock();
        }

        if (i == 0 && og_config.verbose >= 1) {
            float EPS = 1.0 * (stats.tested > 0);
            printf("\nCPU OVERFLOW Timing:\n");
            printf("\ttotal tested   : %lu (%.1f%% CPU, %.1f%% GPU)\n",
                    stats.tested.load(),
                    100.0 * stats.tested_cpu / stats.tested,
                    100.0 * stats.tested_gpu / stats.tested);
            printf("\tspot checked   : %lu (%.6f secs/prob_prime test)\n",
                    stats.spot_checked.load(), stats.d_spot_check / (EPS + stats.spot_checked));
            printf("\tnext prime only: %lu, both sides: %lu\n",
                    stats.skipped_prev.load(), stats.tested_prev.load());
            printf("\ttotal time     : cpu: %.1f, gpu %.1f\n",
                    stats.d_next_prime_cpu.load(), stats.d_next_prime_gpu.load());
            printf("\tnext prime test/sec sieve: %0.f, cpu: %.0f, gpu: %.0f\n",
                    stats.tested / stats.d_next_prime_sieve,
                    stats.tested_cpu / stats.d_next_prime_cpu,
                    stats.tested_gpu / stats.d_next_prime_gpu);
            uint32_t large = stats.greater_than_min_merit;
            if (large) {
                printf("\t> %.1f merit: %u\n", min_merit, large);
                if (stats.pseudoprimes || stats.mismatches) {
                    printf("\tMismatches | Fermat: %lu, Other: %lu\n",
                            stats.pseudoprimes.load(), stats.mismatches.load());
                }
            }
            int32_t missing = stats.tested - stats.tested_cpu - stats.tested_gpu;
            if (missing > 0) {
                printf("\ttests don't add up %lu != %lu + %lu, missing %d\n",
                        stats.tested.load(), stats.tested_cpu.load(),
                        stats.tested_gpu.load(), missing);
            }
            printf("\n");
        }

        if (og_config.verbose >= 3) {
            usleep(i * 10'000); // i * 10ms
            printf("\tCPU overflow(%u) done\n", i);
        }

        mpz_clear(K);
        mpz_clears(center, next_p, prev_p, tmp, tmp2, NULL);
    } catch (const std::exception &e) {
        cout << "ERROR in run_cpu_overflow_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}
