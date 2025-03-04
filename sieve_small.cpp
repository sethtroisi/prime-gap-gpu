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

#include "sieve_small.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>

#include <gmp.h>
#include <omp.h>
#include <primesieve.hpp>

#include "gap_common.h"
#include "cached.h"
#include "sieve_small_gpu.h"


using std::cout;
using std::endl;
using std::map;
using std::mutex;
using std::pair;
using std::vector;
using namespace std::chrono;


// Used to validate factors divide the number claimed
//#define GMP_VALIDATE_FACTORS

class method2_stats {
    public:
        method2_stats() {};

        method2_stats(
                int thread_i,
                const struct Config& config,
                size_t valid_ms,
                double initial_prob_prime
        ) {
            thread = thread_i;

            start_t = high_resolution_clock::now();
            interval_t = high_resolution_clock::now();
            total_unknowns = config.sieve_length * valid_ms;

            next_mult = 1000;

            prob_prime = initial_prob_prime;
            current_prob_prime = prob_prime;
        }

        // Some prints only happen if thread == 0
        int thread = 0;
        uint64_t next_print = 0;
        uint64_t next_mult = 10000;

        // global and interval start times
        high_resolution_clock::time_point  start_t;
        high_resolution_clock::time_point  interval_t;

        long total_unknowns = 0;
        long small_prime_factors_interval = 0;
        // Sum of above two, mostly handled in method2_increment_print
        long prime_factors = 0;

        size_t pi = 0;
        size_t pi_interval = 0;

        uint64_t validated_factors = 0;

        // prob prime after sieve up to some prime threshold
        double current_prob_prime = 0;

        // Constants (more of a stats storage)
        double prob_prime = 0;
        uint64_t last_prime = 0;
};

void method2_increment_print(
        uint64_t prime,
        size_t valid_ms,
        vector<bool> &composite,
        method2_stats &stats,
        const struct Config& config) {

    while (prime >= stats.next_print && stats.next_print < stats.last_prime) {
        //printf("\t\tmethod2_increment_print %'ld >= %'ld\n", prime, stats.next_print);
        const size_t max_mult = 100'000'000'000L * (config.threads > 2 ? 10L : 1L);


        // 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000 ...
        // 60, 70, 80, 90, 100, 120, 150, 200, 300 billion because intervals are wider.
        size_t extra_multiples = prime > ((config.threads > 4) ? 100'000'000 : 1'000'000);
        // With lots of threads small intervals are very fast
        // and large % of time is spent counting unknowns

        // Next time to increment the interval size
        size_t next_next_mult = (5 + 10 * extra_multiples) * stats.next_mult;
        if (stats.next_mult < max_mult && stats.next_print == next_next_mult) {
            stats.next_mult *= 10;
            stats.next_print = 0;
        }

        // 1,2,3,4,5,6,7,8,9,10, SKIP to 12, SKIP to 15
        stats.next_print += stats.next_mult;
        assert(stats.next_print % stats.next_mult == 0);

        if (stats.next_mult < max_mult) {
            int64_t ratio = stats.next_print / stats.next_mult;
            assert(ratio >= 1 && ratio <= 14);

            if (ratio > 10 && ratio < 12) {  // Skip 11 => 12
                stats.next_print = 12 * stats.next_mult;
            } else if (ratio > 12) {  // Skip 13, 14 => 15
                stats.next_print = 15 * stats.next_mult;
            }
        }

        // Never set next_print beyond last_prime
        stats.next_print = std::min(stats.next_print, stats.last_prime);
    }

    bool is_last = (prime == stats.last_prime);

    if (config.verbose + is_last >= 1) {
        auto   s_stop_t = high_resolution_clock::now();
        // total time, interval time
        double     secs = duration<double>(s_stop_t - stats.start_t).count();
        double int_secs = duration<double>(s_stop_t - stats.interval_t).count();
        uint32_t SIEVE_LENGTH = config.sieve_length;

        if (stats.thread >= 1) {
            printf("Thread %d\t", stats.thread);
        }

        stats.pi += stats.pi_interval;

        printf("%'-10ld\t(seconds: %.2f/%-.1f | per m: %.3g)",
            prime,
            int_secs, secs,
            secs / valid_ms);
        if (int_secs > 240) {
            // Add " @ HH:MM:SS" so that it is easier to predict when the next print will happen
            time_t rawtime = std::time(nullptr);
            struct tm *tm = localtime( &rawtime );
            printf(" @ %d:%02d:%02d", tm->tm_hour, tm->tm_min, tm->tm_sec);
        }
        printf("\n");
        stats.interval_t = s_stop_t;

        int verbose = config.verbose + (2 * is_last) + (prime > 1e9) + (stats.thread == 0);
        if (verbose >= 3) {
            stats.prime_factors += stats.small_prime_factors_interval;

            // See THEORY.md
            double prob_prime_after_sieve = stats.prob_prime * log(prime) * exp(GAMMA);
            double delta_sieve_prob = (1/stats.current_prob_prime - 1/prob_prime_after_sieve);
            double skipped_prp = valid_ms * delta_sieve_prob;

            if (is_last || config.threads <= 1) {
                uint64_t t_total_unknowns = std::count(composite.begin(), composite.end(), 0);
                uint64_t new_composites = stats.total_unknowns - t_total_unknowns;

                // count_coprime_sieve * valid_ms also makes sense but leads to smaller numbers
                printf("\tunknowns %'9ld/%-5ld\t"
                       "(avg/m: %.2f) (composite: %.2f%% +%.3f%% +%'ld)\n",
                    t_total_unknowns, valid_ms,
                    1.0 * t_total_unknowns / valid_ms,
                    100.0 - 100.0 * t_total_unknowns / (SIEVE_LENGTH * valid_ms),
                    100.0 * new_composites / (SIEVE_LENGTH * valid_ms),
                    new_composites);

                // NOTE: There used to be some validity check on t_total_unknowns ti required
                // count_coprime_p which is hard to estimate and was directly computed when
                // prime pasted P, this is non-trivial in the GPU only world.

                stats.total_unknowns = t_total_unknowns;
            }

            stats.current_prob_prime = prob_prime_after_sieve;

            double prp_rate = skipped_prp / (int_secs * config.threads);

            printf("\n");

            stats.small_prime_factors_interval = 0;
        }

        stats.pi_interval = 0;
    }
}

/**
 * TODO better name: RangeStats, KStats, Helpers, Indexes?
 * Helper arrays
 */
// This is defined in header but left with comments to better code match here.
/*
class Cached {
    public:
        // mi such that gcd(m_start + mi, D) = 1
        // TODO: Can I change this to m_inc as a uint8_t?
        vector<uint32_t> valid_mi;
        // valid_mi.size()
        size_t valid_ms;

        // m_reindex[mi] = mii (index into composite) if coprime
        // -1 if gcd(ms + i, D) > 1
        //
        // This is potentially very large use is_m_coprime and is_m_coprime2310
        // to pre check it's a coprime mi before doing the L3/RAM lookup.
        vector<int32_t> m_reindex;

        // if gcd(ms + mi, D) = 1
        // TODO try with dynamic_bitset
        vector<bool> is_m_coprime;

        // is_m_coprime2310[i] = (i, D') == 1
        // D' = gcd(2310, D)
        // first 2310 values.
        std::bitset<2310> is_m_coprime2310;

        // X which are coprime to K
        vector<uint32_t> coprime_X;
        // reindex composite[m][X] for composite[m_reindex[m]][x_reindex[X]]
        // Special 0'th entry stands for all not coprime
        vector<uint32_t> x_reindex;

        // reindex composite[m][i] using (m, wheel) (wheel is 1!, 2!, 3!, or 5!)
        // This could be first indexed by x_reindex,
        // Would reduce size from wheel * (SL+1) to wheel * coprime_i

        // Note: Larger wheel eliminates more numbers but takes more space.
        // 6 (saves 2/3 memory), 30 (saves 11/15 memory)
        uint32_t x_reindex_wheel_size;
        vector<uint16_t> x_reindex_wheel; // [x_reindex_wheel_size * SL]
        // x_unindex_wheel[j] = k, where x_reindex_wheel[coprime_X[k]] = j
        vector<uint16_t> x_unindex_wheel // x_reindex_wheel_size * SL]
        vector<size_t> x_reindex_wheel_count; // For stats and packed composite

        uint64_t composite_line_size;

        int32_t K_mod2310;

        // is_comprime2310[i] = (i % 2) && (i % 3) && (i % 5) && (i % 7) && (i % 11)
        vector<char> is_coprime2310;

        Cached(const struct Config& config, const mpz_t &K);
};
*/

Cached::Cached(const struct Config& config, const mpz_t &K) {
        const uint32_t P = config.p;
        const uint32_t D = config.d;

        const uint32_t SL = config.sieve_length;

        const vector<uint32_t> P_primes = get_sieve_primes(P);
        assert( P_primes.back() == P);

        // Allocate temp vectors
        m_reindex.resize(config.minc, -1);
        {
            auto temp = is_coprime_and_valid_m(config);
            is_m_coprime = temp.first;
            valid_mi = temp.second;

            for (uint32_t mii = 0; mii < valid_mi.size(); mii++) {
                uint32_t mi = valid_mi[mii];
                m_reindex[mi] = mii;
            }
        }
        valid_ms = valid_mi.size();

        // Includes 0
        x_reindex.resize(SL+1, 0);

        // reindex composite[m][i] using (m, wheel) (wheel is 1!,2!,3!,5!)
        // Would reduce size from wheel * SL to wheel * coprime_i

        // Note: Larger wheel eliminates more numbers but takes more space.
        // 6 seems reasonable for larger numbers  (uses 1/3 memory = 33%)
        // 30 is maybe better for smaller numbers (uses 4/15 memory = 26%)
        // 210 is maybe better (uses 24/105 = 23% memory) but x_reindex_wheel might not fit in memory.
        uint32_t wheel = gcd(D, METHOD2_WHEEL_MAX);
        uint32_t reindex_size = wheel * SL * sizeof(uint32_t);
        if (reindex_size > 7 * 1024 * 1024) {
            if (wheel % 7) {
                wheel /= 7;
            } else if (wheel % 5) {
                wheel /= 5;
            }
        }
        x_reindex_wheel_size = wheel;
        assert(x_reindex_wheel_size <= METHOD2_WHEEL_MAX); // Static size in Caches will break;

        x_reindex_wheel_count.resize(x_reindex_wheel_size, 0);

        vector<char> is_offset_coprime(SL+1, 1);
        for (uint32_t prime : P_primes) {
            if (D % prime != 0) {
                for (size_t x = 0; x <= SL; x += prime) {
                    is_offset_coprime[x] = 0;
                }
            }
        }
        assert( D % 2 == 0 );
        // 2 | D <=> K is odd <=> all m % 2 == 1
        // m % 2 == 1 -> any odd x will be divisible by two
        // These can't have unknowns so skip by marking as not-coprime.
        // TODO backport this logic assuming it's valid.
        if (1) {
            size_t skipped = 0;
            for (size_t x = 1; x <= SL; x += 2) {
                if (is_offset_coprime[x]) {
                    is_offset_coprime[x] = 0;
                    skipped += 1;
                }
            }
            if (config.verbose >= 2) {
                printf("\tSkipped %lu coprime X which are always divisible by 2\n", skipped);
            }
        }

        // Center should be marked composite by every prime.
        assert(is_offset_coprime[0] == 0);
        {
            size_t coprime_count = 0;
            for (size_t X = 0; X <= SL; X++) {
                if (is_offset_coprime[X] > 0) {
                    coprime_X.push_back(X);
                    coprime_count += 1;
                    x_reindex[X] = coprime_count;
                    //printf("\tcoprime(%lu) = %lu\n", coprime_count, X);
                }
            }
            assert(coprime_count == coprime_X.size());
        }

        // Start at m_wheel == 0 so that re_index_m_wheel == 1 (D=1) works.

        x_reindex_wheel.resize(x_reindex_wheel_size * (SL+1), 0);
        for (size_t m_wheel = 1; m_wheel < x_reindex_wheel_size; m_wheel++) {
            if (gcd(x_reindex_wheel_size, m_wheel) > 1) continue;


            // m * K % wheel => m_wheel % wheel
            uint32_t mod_center = m_wheel * mpz_fdiv_ui(K, x_reindex_wheel_size);

            // 0 is special index
            x_unindex_wheel[m_wheel].push_back(0xFFFF);

            size_t coprime_count_wheel = 0;
            for (size_t i = 0; i < SL+1; i++) {
                if (is_offset_coprime[i] > 0) {
                    if (gcd(mod_center + i, x_reindex_wheel_size) == 1) {
                        coprime_count_wheel += 1;
                        x_reindex_wheel[m_wheel * (SL+1) + i] = coprime_count_wheel;
                        // i is offset, want to know that index in coprime_X
                        x_unindex_wheel[m_wheel].push_back(x_reindex[i] - 1);
                    }
                }
            }
            x_reindex_wheel_count[m_wheel] = coprime_count_wheel;
            assert(coprime_count_wheel >= 1);

            {
                size_t x_reindex_limit = std::numeric_limits<
                    std::remove_extent_t<decltype(x_reindex_wheel)>::value_type>::max();

                // Only happens when P very large (100K)
                // Fix by changing x_reindex_wheel type to int32_t
                assert(coprime_count_wheel < x_reindex_limit);  // See comment above.
            }
        }

        for (const auto X: coprime_X) {
            size_t count = 0;
            for (size_t i = 0; i < x_reindex_wheel_size; i++) {
                if (x_reindex_wheel[i * (SL+1) + X] > 0) {
                    count += 1;
                }
            }
            assert( count > 0 ); // coprime_X which is never coprime???
        }

        uint32_t max_composites = *std::max_element(
                x_reindex_wheel_count.begin(), x_reindex_wheel_count.end());
        composite_line_size = 32 * ((max_composites + 31) / 32);
        if (config.verbose >= 2) {
            cout << "Need at least " << max_composites << " per m, rounding up to "
                 << composite_line_size << endl << endl;;
        }

        K_mod2310 = mpz_fdiv_ui(K, 2310);

        is_coprime2310.resize(2*3*5*7*11, 1);
        for (int p : {2, 3, 5, 7, 11})
            for (size_t i = 0; i < is_coprime2310.size(); i += p)
                is_coprime2310[i] = 0;

        //is_m_coprime2310.resize(2310, 1);
        is_m_coprime2310.set();

        for (int p : {2, 3, 5, 7, 11})
            if (config.d % p == 0)
                for (int i = 0; i < 2310; i += p)
                    is_m_coprime2310[i] = 0;

        assert(count(is_coprime2310.begin(), is_coprime2310.end(), 1) == 480);
        /**
         * Because 2310 is a multiple of x_reindex_wheel_size,
         * x_reindex_wheel will always be true if prefiltered by is_coprime2310[n % 310]
         */
        assert(2310 % x_reindex_wheel_size == 0);
};


std::unique_ptr<SieveOutput> save_unknowns(
        const struct Config& config,
        const mpz_t &K,
        const Cached &caches,
        vector<bool> &composite) {
    // For 50M range with 5M sieve, this took 3.9 of 74 seconds!
    // For 10M range with 1M sieve, this took 0.8 of 11 seconds!
    // For 20M range with 1M sieve, this took 1.7 of 23 seconds!
    auto s_save_t = high_resolution_clock::now();

    const uint64_t M_start = config.mstart;
    const uint32_t D = config.d;
    const int32_t SL = config.sieve_length;

    uint32_t count_by_X[SL+1] = {};

    size_t count_a = 0;
    size_t count_b = caches.valid_mi.size() * caches.coprime_X.size();

    auto output = std::make_unique<SieveOutput>(M_start, SL);
    {
        output->coprime_X.reserve(caches.coprime_X.size());
        for (uint32_t x : caches.coprime_X) {
            // Limits sieve_length to 32K
            assert(x < 0x8FFF);
            output->coprime_X.push_back(x);
        }
    }

    size_t count_m = caches.valid_ms;
    output->m_inc.reserve(count_m);
    output->unknowns.resize(count_m);

    // Create all vectors so that threads don't need to be ordered.
    {
        uint64_t m_last = M_start;

        for (uint64_t mi : caches.valid_mi) {
            uint64_t m = M_start + mi;
            assert(gcd(m, D) == 1);
            uint16_t delta = m - m_last;
            assert( delta <= 0x7F );
            output->m_inc.emplace_back(m - m_last, /* found */ 0);
            m_last = m;
        }
    }

    #pragma omp parallel for ordered schedule(dynamic, 8) num_threads(config.threads) reduction(+:count_a)
    for (size_t mii = 0; mii < count_m; mii++) {
        uint64_t mi = caches.valid_mi[mii];
        uint64_t m = M_start + mi;
        assert((signed)mii == caches.m_reindex[mi]);

        const size_t composite_index = mii * caches.composite_line_size;
        const uint32_t m_mod_wheel = m % caches.x_reindex_wheel_size;
        // TODO test with and without &
        const auto x_reindex_m = caches.x_reindex_wheel.data() + ((m_mod_wheel) * (SL+1));
        //assert(x_reindex_m.size() == (uint64_t) (SL + 1));
        const auto &x_unindex_m = caches.x_unindex_wheel[m_mod_wheel];

        int64_t found = 0;
        auto& deltas = output->unknowns[mii];
        // threadlocal char tmp[1000], didn't speed this up, but maybe would reduce memory
        //deltas.reserve(found);

        // TODO consider if I can use __builtin_ctz or __builtin_ffs to avoid looking at each index
        // would take 8 bytes from comp vector and make a 64bit int then do repeat builtin_ffs.

        // TODO uint8 delta is probably valid again!

        // Index of last unknown (in coprime_X)
        int last_u_i = 0;

        const size_t max_i = composite_index + caches.composite_line_size;
        for (size_t i = composite_index; i < max_i; i++) {
            if (composite[i] == 1) continue;

            size_t j = i - composite_index;
            auto u_i = x_unindex_m[j];
            assert( x_reindex_m[caches.coprime_X[u_i]] == j );

            int delta = u_i - last_u_i;

            assert( delta <= 0xFFFF );
            deltas.push_back(delta);
            count_by_X[caches.coprime_X[u_i]]++;

            last_u_i = u_i;
            found += 1;
        }

        assert( found <= 0xFF );
        std::get<1>(output->m_inc[mii]) = found;
        count_a += found;
    }

    {
        // Every unknown is at a coprime. This is a tautology from count_by_X[...]
        for(int32_t i = 0; i <= SL; i++) {
            if (count_by_X[i] > 0) {
                assert( caches.x_reindex[i] > 0 );
                assert( caches.coprime_X[caches.x_reindex[i] - 1] == i );
            }
        }

        if (caches.valid_ms >= 100'000) {
            // Every coprime should have an unknown
            for (const auto X : caches.coprime_X) {
                if (count_by_X[X] == 0) {
                    cout << "\tNo unknowns for " << X << endl;
                }
                assert( count_by_X[X] > 0 );
            }
        }
    }

    if (config.verbose >= 0) {
        auto s_stop_t = high_resolution_clock::now();
        std::string fn = Args::gen_unknown_fn(config, ".txt");
        printf("\n\tSaved deltas for '%s'\n", fn.c_str());
        printf("\t\t%ld/%ld (%.1f%%) -> %.1f/m (%.1f%% of SL) in %.1f seconds\n\n",
               count_a, count_b, 100.0f * count_a / count_b,
               1.0f * count_a / count_m, (100.0f * count_a / count_m / SL),
               duration<double>(s_stop_t - s_save_t).count());
    }

    return output;
}


std::unique_ptr<SieveOutput> prime_gap_parallel(const struct Config& config) {
    // Method2
    const uint64_t M_start = config.mstart;
    const uint32_t M_inc = config.minc;

    const uint32_t P = config.p;

    const uint32_t SIEVE_LENGTH = config.sieve_length;

    const uint64_t MAX_PRIME = config.max_prime;

    const uint64_t LAST_PRIME = [&] {
        mpz_t test;
        mpz_init(test);

        mpz_set_ui(test, MAX_PRIME);
        mpz_prevprime(test, test);

        uint64_t temp = mpz_get_ui(test);
        mpz_clear(test);
        return temp;
    }();

    assert( LAST_PRIME <= MAX_PRIME && LAST_PRIME + 500 > MAX_PRIME);

    // ----- Generate primes for P
    const vector<uint32_t> P_primes = get_sieve_primes(P);
    assert( P_primes.back() == P);

    // ----- Sieve stats & Merit Stuff
    mpz_t K;
    const double K_log = prob_prime_and_stats(config, K);
    const double N_log = K_log + log(config.mstart);
    const double prob_prime = 1 / N_log - 1 / (N_log * N_log);


    // ----- Allocate memory

    // Various pre-calculated arrays of is_coprime arrays
    const Cached caches(config, K);
    const size_t valid_ms = caches.valid_ms;
    const uint32_t x_reindex_wheel_size = caches.x_reindex_wheel_size;

    const size_t count_coprime_sieve = caches.coprime_X.size();

    // TODO using P+1 is faster, but not sure if worth CPU vs GPU trade-off
    if (config.verbose >= 1) {
        printf("sieve_length:    %'d\n", config.sieve_length);
        printf("max_prime:       %'ld\n", config.max_prime);
    }


    // ----- Timing
    if (config.verbose >= 2) {
        printf("\n");
    }

    /**
     * Much space is saved via a reindexing scheme
     * composite[mi][x] (0 <= mi < M_inc, 0 <= x <= SL) is re-indexed to
     *      composite[m_reindex[mi]][x_reindex_wheel[m%wheel_size][x]]
     * m_reindex[mi] with (D, M + mi) > 0 are mapped to -1 (and must be handled by code)
     * x_reindex[x]  with (K, x) > 0 are mapped to 0 (and that bit is ignored)
     * x_reindex_wheel[x] same as x_reindex[x]
     */

    vector<bool> composite(valid_ms * caches.composite_line_size);
    {
        int align_print = 0;
        /**
         * Per m_inc
         *      4 bytes in m_reindex
         *      1 bit   in is_m_coprime
         * Per valid_ms
         *      4  bytes in caches.valid_mi (could be 1 byte if valid_mi -> m_inc);
         *      40 bytes for vector<bool> instance
         *      count_coprime_sieve + 1 bits
         */

        size_t MB = 8 * 1024 * 1024;
        size_t overhead_bits = M_inc * (8 * sizeof(uint32_t) + 1) +
                               valid_ms * 8 * sizeof(uint32_t) +
                               composite.size();

        // Per valid_ms
        size_t guess = overhead_bits + valid_ms * (count_coprime_sieve + 1);
        if (config.verbose >= 1) {
            // Using strings instead of printf so sizes can be aligned.
            std::string s_coprime_m = "coprime m    " +
                std::to_string(valid_ms) + "/" + std::to_string(M_inc) + " ";
            std::string s_coprime_i = "coprime i    " +
                std::to_string(count_coprime_sieve) + "/" + std::to_string(SIEVE_LENGTH);
            align_print = s_coprime_m.size();

            printf("%*s", align_print + (int) s_coprime_i.size(), "");
            printf("  ~%'ld MB overhead\n", overhead_bits / MB);
            printf("%s%s, ~%'ld MB\n", s_coprime_m.c_str(), s_coprime_i.c_str(), guess / MB);
        }

        if (x_reindex_wheel_size > 1) {
            // Update guess with first wheel count for OOM prevention check
            size_t guess_avg_count_coprime = caches.x_reindex_wheel_count[1];
            guess = overhead_bits + valid_ms * (guess_avg_count_coprime + 1);
        }

        // Try to prevent OOM, check composite < 10GB allocation,
        if (guess > (size_t) config.max_mem * 1024 * MB) {
            if (config.verbose >= 1 && x_reindex_wheel_size > 1) {
                size_t allocated = guess - overhead_bits;
                printf("%*s", align_print, "");
                printf("coprime wheel %ld/%d, ~%'ldMB\n",
                    allocated / valid_ms, SIEVE_LENGTH,
                    allocated / 8 / 1024 / 1024);
            }
            printf("\ncombined_sieve expects to use %'ld MB which is greater than %d GB limit\n",
                    guess / MB, config.max_mem);
            printf("\nAdd `--max-mem %ld` to skip this warning\n", (guess / 1024 / MB) + 1);
            exit(1);
        }

        size_t allocated = composite.size();
        for (size_t i = 0; i < valid_ms; i++) {
            int m_wheel = (M_start + caches.valid_mi[i]) % x_reindex_wheel_size;
            assert(gcd(m_wheel, x_reindex_wheel_size) == 1);

            composite[i * caches.composite_line_size] = true;
            // disable all the extra padding bits
            size_t used = caches.x_reindex_wheel_count[m_wheel] + 1;
            for (size_t j = used; j < caches.composite_line_size; j++) {
                composite[i * caches.composite_line_size + j] = true;
            }
        }
        if (config.verbose >= 1 && x_reindex_wheel_size > 1) {
            printf("%*s", align_print, "");
            printf("coprime wheel %ld/%d, ~%'ld MB\n",
                allocated / valid_ms, SIEVE_LENGTH,
                allocated / 8 / 1024 / 1024);
        }

        if (config.verbose >= 1) {
            cout << "\tcomposite line: " << caches.composite_line_size
                << " total: " << composite.size() << endl;
            printf("\n");
        }
    }

    // Used for various stats
    method2_stats stats(/* thread */ 0, config, valid_ms, prob_prime);
    stats.last_prime = LAST_PRIME;

    {
        auto gsieve = GPUSieve(config, K, caches, config.max_prime);
        gsieve.run_sieve(config, config.mstart, config.minc, caches, composite);
    }

    method2_increment_print(LAST_PRIME, caches.valid_ms, composite, stats, config);

    auto result = save_unknowns(config, K, caches, composite);

    mpz_clear(K);

    return result;
}
