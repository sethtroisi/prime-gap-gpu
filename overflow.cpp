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
#include "gpu_testing.h"


using std::cout;
using std::cerr;
using std::endl;
using namespace std::chrono;

/** Extern globals */

OverflowQueue overflow;

/** Tuning Parameters */

const uint32_t OVERFLOW_SIEVE_LIMIT = 150'000;
const bool EXTRA_CHECKS = false;

/** Globals for this class */

std::mutex record_mtx;

class TestingStats {
    public:
        std::atomic<uint64_t> tested = 0;
        std::atomic<uint64_t> tested_cpu = 0;
        std::atomic<uint64_t> tested_gpu = 0;

        // Set to the largest m computed during execution
        std::atomic<uint64_t> max_m = 0;

        std::atomic<uint64_t> skipped_prev = 0;
        std::atomic<uint64_t> tested_prev = 0;

        std::atomic<uint64_t> greater_than_min_merit = 0;
        // GAP MISMATCH where next_prime had pow(2, n-1, n) == 1
        std::atomic<uint64_t> pseudoprimes = 0;
        // GAP MISMATCH where ^ is NOT TRUE.
        std::atomic<uint64_t> mismatches = 0;

        std::atomic<uint64_t> spot_checked = 0;

        std::atomic<double>   d_sieve{0.0};
        std::atomic<double>   d_next_prime_cpu{0.0};
        std::atomic<double>   d_next_prime_gpu{0.0};
        std::atomic<double>   d_next_prime_gpu_misc{0.0};
        std::atomic<double>   d_prev_prime_cpu{0.0};
        std::atomic<double>   d_spot_check{0.0};
};


class OverflowMisc {
    public:
        OverflowMisc(
                vector<uint16_t> &cX,
                vector<uint16_t> &cin,
                vector<std::pair<uint32_t, uint32_t>> &panrs,
                vector<std::pair<uint32_t, uint32_t>> &panr,
                uint64_t d,
                uint64_t kmd,
                vector<uint16_t> &dw, vector<uint16_t> &dwn) :
            coprime_X(cX), coprime_index_next(cin),
            p_and_neg_r_small(panrs), p_and_neg_r(panr),
            D(d), K_mod_d(kmd),
            d_wheel(dw), d_wheel_next(dwn) {};

        OverflowMisc() {};

        vector<uint16_t> coprime_X;
        // coprime_X[coprime_index_next[t]] >= t
        vector<uint16_t> coprime_index_next;
        vector<std::pair<uint32_t, uint32_t>> p_and_neg_r_small;
        vector<std::pair<uint32_t, uint32_t>> p_and_neg_r;

        uint64_t D;
        uint64_t K_mod_d;
        vector<uint16_t> d_wheel;
        // d_wheel[d_wheel_next[t]] >= t
        vector<uint16_t> d_wheel_next;
} ofs;


void setup_overflow(const struct Config config) {
    vector<uint16_t> coprime_X;
    vector<uint16_t> coprime_index_next;
    vector<std::pair<uint32_t, uint32_t>> p_and_neg_r_small;
    vector<std::pair<uint32_t, uint32_t>> p_and_neg_r;

    mpz_t K;
    init_K(config, K);

    // 10 -> 3% overflow, 21K sieves / second
    // 11 -> 1% overflow, 20K sieves / second
    // 15 -> .5% overflow, 18K sieves / second
    uint32_t stop_x = 11 * config.p;
    {
        auto X = get_coprime_X(config, stop_x);
        coprime_X.reserve(X.size());
        for (const auto x : X) {
            coprime_X.push_back(x);
        }

        coprime_index_next.resize(coprime_X.back() + 1, 0xFFFF);
        uint16_t i = 0;
        for (uint16_t x_i = 0; x_i < coprime_X.size(); x_i++) {
            for ( ; i <= coprime_X[x_i]; i++ ) {
                coprime_index_next[i] = x_i;
            }
        }
        for (i = 0; i < coprime_index_next.size(); i++) {
            assert( coprime_X[coprime_index_next[i]] >= i );
        }
    }

    uint64_t D = config.d;
    uint64_t K_mod_d = mpz_fdiv_ui(K, D);
    assert( 1 <= K_mod_d && K_mod_d < D );

    vector<uint16_t> d_wheel;
    vector<uint16_t> d_wheel_next;
    {
        assert(D < 65000); // Not as useful otherwise
        for (uint32_t i = 1; i < D; i++) {
            if (gcd(i, D) != 1)
                d_wheel.push_back(i);
        }

        // Default value is d_wheel.size();
        // Include one extra value so mod+1 is possible
        d_wheel_next.resize(D+1, d_wheel.size());

        uint16_t i = 0;
        for (uint16_t d_i = 0; d_i < d_wheel.size(); d_i++) {
            for ( ; i <= d_wheel[d_i]; i++ ) {
                d_wheel_next[i] = d_i;
            }
        }

        for (i = 0; i < d_wheel_next.size(); i++) {
            uint16_t d_i = d_wheel_next[i];
            assert( d_i == d_wheel.size() || d_wheel[d_wheel_next[i]] >= i );
        }
    }


    {
        primesieve::iterator iter;
        uint64_t prime = iter.next_prime();
        assert (prime == 2);  // we skip 2 which is the oddest prime.
        for (prime = iter.next_prime(); prime < OVERFLOW_SIEVE_LIMIT; prime = iter.next_prime()) {
            // factors of D handled by d_wheel
            if (prime <= config.p)
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

    ofs = OverflowMisc{
        coprime_X, coprime_index_next,
        p_and_neg_r_small, p_and_neg_r,
        D, K_mod_d, d_wheel, d_wheel_next};

    mpz_clear(K);
}


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
        mpz_sub(tmp, next_p, prev_p);
        uint64_t test_gap = mpz_get_ui(tmp);
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
                gmp_printf("%Zd\n", next_p);
                printf("\tGAP MISMATCH! %lu vs %lu at %lu * %u# / %u + %lu\n",
                        test_gap, gap, m, P, D, test_gap - prev_gap);
            }
            gap = test_gap;
            merit = test_gap / (K_log + log(m));
        }

        if (merit > min_merit) {
            std::string record = std::format(
                    "{} {:.3f} {} * {}# / {} - {}",
                    gap, merit, m, P, D, prev_gap);
            cout << record << endl;

            record_mtx.lock();
            record_stream << record << endl;
            record_stream.flush();
            record_mtx.unlock();
        }
    }
}

/** Expects center to be correctly set */
static
void handle_next_prime_result(
        const uint32_t MIN_GAP_TO_CONTINUE,
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

    if (EXTRA_CHECKS) {
        mpz_mul_ui(tmp2, K, m);
        assert( mpz_cmp(tmp2, center) == 0);
    }

    auto s_start_t = high_resolution_clock::now();
    mpz_prevprime(prev_p, center);
    mpz_sub(prev_p, center, prev_p);
    uint64_t prev_gap = mpz_get_ui(prev_p);
    uint64_t gap = prev_gap + next_gap;
    double merit = gap / (K_log + log(m));
    stats.d_prev_prime_cpu += duration<double>(high_resolution_clock::now() - s_start_t).count();

    process_result(
        min_merit, K_log, P, D, K, center,
        m, merit, gap, prev_gap,
        next_p, prev_p, tmp, tmp2, stats,
        record_stream);
}

/**
 * [sieve_start, sieve_start + sieve_length)
 * If not is_positive [-sieve_start, -sieve_start - sieve_length)
 */
static
void sieve_interval_cpu(const uint64_t m,
        const bool is_positive,
        const uint32_t sieve_start,
        const uint32_t sieve_length,
        vector<uint8_t> &composite,
        TestingStats &stats,
        mpz_t &tmp, const mpz_t &K
) {

    auto s_start_t = high_resolution_clock::now();

    // TODO stop storing evens.
    uint16_t bytes = (sieve_length + 7) / 8 + 1;
    composite.resize(bytes, 0);
    std::fill(composite.begin(), composite.end(), 0);

    // only interested in even i
    assert(sieve_start % 2 == 0);
    // otherwise m * neg_r < sieve_start
    assert(m > ofs.p_and_neg_r.back().first);

    // Otherwise I need to do something different here
    // Technically should check m * K_mod_d < 60 bits
    assert(std::log2(m) + std::log2(ofs.p_and_neg_r.back().first) < 60);

    uint64_t D = ofs.D;

    // Tile d_wheel into composite with a possible offset
    if (is_positive) {
        uint64_t wheel_start = (m * ofs.K_mod_d + sieve_start) % D;
        uint32_t w_n = ofs.d_wheel.size();
        uint32_t w_i = ofs.d_wheel_next[wheel_start];
        assert( w_i == w_n || ofs.d_wheel[w_i] >= wheel_start);
        assert( w_i == 0   || ofs.d_wheel[w_i-1] < wheel_start);

        // Technically wheel_start is always odd, but we only care about even t
        // could only use odd valued d_wheel.
        for (int32_t j = -wheel_start; j < (signed) sieve_length; ) {
            for ( ; w_i < w_n; w_i++) {
                int32_t t = j + ofs.d_wheel[w_i];
                assert( t >= 0 );
                if (t > (signed) sieve_length)
                    break;
                composite[t >> 3] |= 1 << (t & 7);
            }
            j += D;
            w_i = 0;
        }
    } else {
        assert( sieve_start == 0);
        uint64_t wheel_start = m * ofs.K_mod_d % D;
        uint32_t w_n = ofs.d_wheel.size();
        // Looking for first number <= wheel_start
        int32_t w_i = ofs.d_wheel_next[wheel_start+1];
        assert( w_i == (signed) w_n || ofs.d_wheel[w_i+1] > wheel_start);
        w_i -= 1;
        assert( w_i < 0 || ofs.d_wheel[w_i] <= wheel_start);

        // composite[0] is wheel_start
        // composite[1] is wheel_start - 1
        for (int32_t j = wheel_start; j < (signed) sieve_length; ) {
            for ( ; w_i >= 0; w_i--) {
                int32_t t = j - ofs.d_wheel[w_i];
                assert( t >= 0 );
                if (t > (signed) sieve_length)
                    break;
                composite[t >> 3] |= 1 << (t & 7);
                if (EXTRA_CHECKS) {
                    mpz_mul_ui(tmp, K, m);
                    mpz_sub_ui(tmp, tmp, t);
                    mpz_t g;
                    mpz_init(g);
                    if ( mpz_gcd_ui(g, tmp, D) == 1) {
                        gmp_printf("(%lu*K - %u, D) = %Zd | %lu -> %d - %u\n",
                                m, t, g, wheel_start, j, ofs.d_wheel[w_i]);
                    }
                    assert( mpz_gcd_ui(g, tmp, D) > 1 );
                    mpz_clear(g);
                }
            }
            j += D;
            w_i = w_n - 1;
        }
    }

    if (is_positive) {
        for (const auto& [p, neg_r] : ofs.p_and_neg_r_small) {
            // -(m * K + sieve_start) % r
            uint64_t temp = m * neg_r - sieve_start;
            uint64_t center_mod = temp % ((uint64_t) p);
            center_mod += (center_mod & 1) ? p : 0;

            uint32_t two_p = p << 1;
            for (uint32_t i = center_mod; i < sieve_length; i += two_p) {
                composite[i >> 3] |= 1 << (i & 7);
            }
        }

        for (const auto& [p, neg_r] : ofs.p_and_neg_r) {
            // -(m * K + sieve_start) % r
            uint64_t temp = m * neg_r - sieve_start;
            uint64_t center_mod = temp % ((uint64_t) p);
            if (center_mod < sieve_length && (center_mod & 1) == 0) {
                composite[center_mod >> 3] |= 1 << (center_mod & 7);
            }
        }
    } else {
        for (const auto& [p, neg_r] : ofs.p_and_neg_r_small) {
            // (m * K - sieve_start) % r
            uint64_t temp = m * (p - neg_r) + sieve_start;
            uint64_t center_mod = temp % ((uint64_t) p);
            center_mod += (center_mod & 1) ? p : 0;

            uint32_t two_p = p << 1;
            for (uint32_t i = center_mod; i < sieve_length; i += two_p) {
                composite[i >> 3] |= 1 << (i & 7);
            }
        }

        for (const auto& [p, neg_r] : ofs.p_and_neg_r) {
            // (m * K - sieve_start) % r
            uint64_t temp = m * (p - neg_r) + sieve_start;
            uint64_t center_mod = temp % ((uint64_t) p);
            if (center_mod < sieve_length && (center_mod & 1) == 0) {
                composite[center_mod >> 3] |= 1 << (center_mod & 7);
            }
        }
    }

    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
    stats.d_sieve += total_s;
}

static
uint32_t next_prime_distance(
        const uint64_t m, const uint32_t min_x,
        const mpz_t &K, mpz_t &center, mpz_t &tmp,
        vector<uint8_t> &composite_tmp,
        TestingStats &stats) {

    mpz_mul_ui(center, K, m);
    mpz_add_ui(tmp, center, min_x);

    if (min_x + 500 < ofs.coprime_X.back()) {
        uint32_t min_x_i = ofs.coprime_index_next[min_x];
        uint32_t next_possible_x = ofs.coprime_X[min_x_i];

        assert( 1 <= min_x_i && min_x_i < ofs.coprime_X.size() );
        assert( min_x <= next_possible_x );
        assert( ofs.coprime_X[min_x_i-1] < min_x );

        sieve_interval_cpu(
            m, true, next_possible_x, ofs.coprime_X.back() - next_possible_x + 1,
            composite_tmp, stats, tmp, K);

        const uint32_t N = ofs.coprime_X.size();
        for (uint32_t x_i = min_x_i; x_i < N; x_i++) {
            uint16_t x = ofs.coprime_X[x_i];
            uint16_t j = x - next_possible_x;
            if ((composite_tmp[j >> 3] & (1 << (j & 7))) == 0) {
                mpz_add_ui(tmp, center, x);
                if (mpz_probab_prime_p(tmp, 20)) {
                    return x;
                }
            }
        }
        mpz_add_ui(tmp, center, ofs.coprime_X.back());
    }

    // Fallback to mpz_nextprime if very large
    mpz_nextprime(tmp, tmp);
    mpz_sub(tmp, tmp, center);
    return mpz_get_ui(tmp);
}


static
void run_tests_on_cpu(
        const uint32_t MIN_GAP_TO_CONTINUE, const float min_merit,
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


class OverflowBatch {
    public:
        const uint32_t N = 4096;

        // TODO parametrize this number.
        GPUBatch gpu_batch{N};

        // Start looking for a non-active entry here
        size_t i = 0;
        size_t added = 0;

        // m, current coprime_X index, sieve_start, next_gap
        // if next_gap == 0, finding next_prime, if > 0 finding prev_prime
        vector<std::tuple<uint64_t, uint16_t, uint16_t, uint16_t>> data;
        // Optimized for less handling, could be 10x smaller by changing to bitset over coprime_x.
        vector<vector<uint8_t>> composite_tmp;

        OverflowBatch()  {
            size_t n = gpu_batch.m_i.size();
            composite_tmp.resize(n);
            data.resize(n);
        }

        void remove_entry(size_t j) {
            gpu_batch.active[j] = 0;
            added--;
            if (j < i) {
                i = j;
            }
        }

        OverflowBatch(const OverflowBatch&) = delete;
        OverflowBatch& operator=(const OverflowBatch&) = delete;
};


static
void push_to_overflow_batch(
        OverflowBatch &overflow_batch,
        const uint64_t m, const uint32_t min_x_i, const uint32_t sieve_start,
        const mpz_t &K, mpz_t &center,
        vector<uint8_t> &composite_tmp,
        TestingStats &stats) {
    auto s_start_t = high_resolution_clock::now();

    GPUBatch &gpu_batch = overflow_batch.gpu_batch;

    uint32_t i = overflow_batch.i; // start search here.
    for (; i < overflow_batch.N; i++) {
        if (gpu_batch.active[i] == 0) {
            break;
        }
    }

    mpz_mul_ui(center, K, m);

    overflow_batch.i = i+1;
    overflow_batch.added++;
    assert(i < overflow_batch.N);
    assert(gpu_batch.active[i] == 0);
    overflow_batch.gpu_batch.active[i] = true;
    mpz_add_ui(*overflow_batch.gpu_batch.z[i], center, sieve_start);
    assert( sieve_start == ofs.coprime_X[min_x_i] );

    overflow_batch.data[i] = {m, min_x_i, sieve_start, 0};
    overflow_batch.composite_tmp[i].swap(composite_tmp);

    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
    stats.d_next_prime_gpu_misc += total_s;
}

static
uint32_t run_overflow_batch(
        GPURunner &runner,
        OverflowBatch &overflow_batch,
        const uint32_t MIN_GAP_TO_CONTINUE,
        const float MIN_MERIT, const float K_log, const uint32_t P, const uint32_t D,
        const mpz_t &K, mpz_t &center, mpz_t &next_p, mpz_t &prev_p,
        mpz_t &tmp, mpz_t& tmp2,
        TestingStats &stats, std::ofstream &record_stream) {

    GPUBatch &gpu_batch = overflow_batch.gpu_batch;
    for (uint32_t i = 0; i < overflow_batch.N; i++) {
        // May need to disable for last batch
        assert( gpu_batch.active[i] == 1 );
    }

    gpu_batch.i = overflow_batch.added;
    std::fill(gpu_batch.result.begin(), gpu_batch.result.end(), -1);

    // Run gpu_batch on GPU.
    auto s_start_t = high_resolution_clock::now();
    runner.run( gpu_batch );
    stats.d_next_prime_gpu += duration<double>(high_resolution_clock::now() - s_start_t).count();

    // Process results.
    s_start_t = high_resolution_clock::now();
    uint32_t finished_items = 0;
    for (uint32_t i = 0; i < overflow_batch.N; i++) {
        assert( gpu_batch.active[i] );
        assert (gpu_batch.result[i] == 0 || gpu_batch.result[i] == 1);
        auto [m, x_i, sieve_start, next_gap] = overflow_batch.data[i];
        uint32_t prev_gap = 0;
        auto is_next_prime = (m > 0);
        m = is_next_prime ? m : -m;

        bool remove = false;
        bool change_to_prev = false;
        bool process = false;

        if (gpu_batch.result[i] == 1) {
            // Found prime for m!
            if (next_gap == 0) {
                change_to_prev = true;
                next_gap = ofs.coprime_X[x_i];

                if (EXTRA_CHECKS) {
                    mpz_mul_ui(center, K, m);
                    mpz_add_ui(tmp, center, next_gap);
                    assert( mpz_cmp(tmp, *gpu_batch.z[i]) == 0 );
                }
            } else {
                remove = true;
                process = true;
                prev_gap = ofs.coprime_X[x_i];

                if (EXTRA_CHECKS) {
                    mpz_mul_ui(center, K, m);
                    mpz_sub_ui(tmp, center, prev_gap);
                    assert( mpz_cmp(tmp, *gpu_batch.z[i]) == 0 );
                }
            }
        } else {
            // Advance to next test see `next_prime_distance`
            uint32_t last_x = ofs.coprime_X[x_i];

            uint32_t M = ofs.coprime_X.size();
            x_i++;
            for (; x_i < M; x_i++) {
                uint16_t x = ofs.coprime_X[x_i];
                uint16_t j = x - sieve_start;
                if ((overflow_batch.composite_tmp[i][j >> 3] & (1 << (j & 7))) == 0) {
                    if (next_gap == 0) {
                        mpz_add_ui(*gpu_batch.z[i], *gpu_batch.z[i], x - last_x);
                    } else {
                        mpz_sub_ui(*gpu_batch.z[i], *gpu_batch.z[i], x - last_x);
                    }
                    // Write back updated x_i;
                    std::get<1>(overflow_batch.data[i]) = x_i;
                    break;
                }
            }

            if (x_i >= M) {
                uint32_t min_x = ofs.coprime_X.back() + 1;

                if (next_gap == 0) {
                    change_to_prev = true;

                    auto s_start_t = high_resolution_clock::now();
                    // Fallback to mpz_nextprime when > coprime_X[-1]
                    mpz_mul_ui(center, K, m);
                    mpz_add_ui(tmp, center, min_x);
                    mpz_nextprime(tmp, tmp);
                    mpz_sub(tmp, tmp, center);

                    next_gap = mpz_get_ui(tmp);
                    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
                    stats.d_next_prime_cpu += total_s;
                    stats.tested_cpu += 1;
                } else {
                    process = true;
                    remove = true;

                    auto s_start_t = high_resolution_clock::now();
                    // Fallback to mpz_prevprime when > coprime_X[-1]
                    mpz_mul_ui(center, K, m);
                    mpz_sub_ui(tmp, center, min_x);
                    mpz_prevprime(tmp, tmp);
                    mpz_sub(tmp, center, tmp);

                    prev_gap = mpz_get_ui(tmp);
                    double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
                    stats.d_next_prime_cpu += total_s;
                    stats.tested_cpu += 1;
                }
            }
        }

        if (process) {
            assert( remove );
            assert( next_gap > 0 && prev_gap > 0 );
            uint64_t gap = prev_gap + next_gap;
            double merit = gap / (K_log + log(m));

            if (EXTRA_CHECKS) {
                mpz_mul_ui(center, K, m);
                mpz_sub_ui(tmp, center, prev_gap);
                assert( mpz_probab_prime_p(tmp, 20) );
                mpz_add_ui(tmp, center, next_gap);
                assert( mpz_probab_prime_p(tmp, 20) );
            }

            stats.tested_prev++;
            process_result(
                MIN_MERIT, K_log, P, D, K, center,
                m, merit, gap, prev_gap,
                next_p, prev_p, tmp, tmp2, stats, record_stream);
        }

        if (change_to_prev) {
            stats.tested_gpu++;
            assert( !remove );
            if (next_gap < MIN_GAP_TO_CONTINUE) {
                stats.skipped_prev++;
                remove = true;
            } else {
                uint32_t x_0 = ofs.coprime_X.front();
                sieve_interval_cpu(
                    m, false, 0, ofs.coprime_X.back() + 1,
                    overflow_batch.composite_tmp[i], stats, tmp, K);

                // Reset this range to be prev_prime search
                overflow_batch.data[i] = {m, 0, 0, next_gap};
                mpz_mul_ui(center, K, m);
                mpz_sub_ui(*gpu_batch.z[i], center, x_0);
            }
        }

        if (remove) {
            overflow_batch.remove_entry(i);
            finished_items++;
        }
    }
    stats.d_next_prime_gpu_misc += duration<double>(high_resolution_clock::now() - s_start_t).count();

    //printf("\tRan overflow on GPU found: %lu primes\n", overflow_batch.N - overflow_batch.added);
    return finished_items;
}


void run_cpu_overflow_worker(const int thread_index,
                             const struct Config og_config,
                             TestingStats &stats) {
    {
        std::string name = std::format("CPU_WORKER_{}", thread_index);
        pthread_setname_np(pthread_self(), name.c_str());
        std::ignore = nice(+10);
    }
    mpz_t K, center, next_p, prev_p, tmp, tmp2;
    mpz_inits(center, next_p, prev_p, tmp, tmp2, NULL);

    const uint32_t P = og_config.p;
    const uint32_t D = og_config.d;

    if (thread_index == 0) {
       prob_prime_and_stats(og_config, K);
    } else {
       init_K(og_config, K);
    }
    double K_log = _log(K);

    const float MIN_MERIT = og_config.min_merit;
    // 2-5x what comes in per batch
    const uint64_t overflow_too_much = og_config.m_inc * og_config.cpu_fraction;

    // See THEORY.md! +1 is optimal-ish +1.XX is small preference for doing less prev_p.
    const float MIN_MERIT_TO_CONTINUE = 1.95 + std::log2(MIN_MERIT * std::log(2) + 1);
    const float m_log = log(og_config.m_start + og_config.m_inc);
    const uint32_t MIN_GAP_TO_CONTINUE =  MIN_MERIT_TO_CONTINUE * (K_log + m_log);

    if (thread_index == 0 && og_config.verbose >= 1) {
        setlocale(LC_NUMERIC, "");
        // ----- Merit / Sieve stats
        float m_log = log(og_config.m_inc);
            printf("Min Gap ~= %'d (for merit > %.1f)\n",
                (int) (MIN_MERIT * (K_log + m_log)), MIN_MERIT);
            printf("Min Gap to continue ~= %'d (merit = %.1f)\n",
                   MIN_GAP_TO_CONTINUE, MIN_MERIT_TO_CONTINUE);
        setlocale(LC_NUMERIC, "C");
    }

    std::ofstream record_stream(std::format("records_{}.txt", P), std::ios_base::app);

    GPURunner runner{};
    OverflowBatch overflow_batch{};
    vector<uint8_t> composite_tmp;
    uint64_t max_m = 0;

    while (is_running) {
        // Wait till size is not zero
        overflow.size.wait(0);

        if (!is_running) {
            break;
        }
        overflow.lock();
        if (overflow.size == 0) {
            overflow.unlock();
            continue; // Might have been removed while locking.
        }

        if (stats.tested % 100'000 == 0 && overflow.size > overflow_too_much) {
            printf("\tCPU Sieve Queue is behind: %u open, %lu processed\n",
                    overflow.size.load(), stats.tested.load());
        }

        // Maybe move this to the helper?
        if (stop_queue > 0) {
            uint32_t rem = overflow.size;
            bool is_power_print = false;
            for (uint64_t p = 1000; p <= rem; p *= 10) {
                is_power_print |= (rem == p) || (rem == 2*p) || (rem == 5*p);
            }
            if (is_power_print) {
                printf("\tFinalizing(stage %d): %u open, %lu processed\n",
                    stop_queue.load(), rem,
                    stats.tested.load());
            }
        }

        assert( overflow.queue.size() == overflow.size );

        auto [m, d, type] = overflow.queue.front(); overflow.queue.pop_front();
        overflow.size--;

        if ( type == Overflow::Type::NEXT_PRIME ) {
            // Do this immediatly to prevent prints (above) with same tested value.
            stats.tested++;
        }
        overflow.unlock();

        if ( type == Overflow::Type::STOP_WORKER ) {
            break;
        }

        // Handle SPOT_CHECK DIRECTLY
        if ( type == Overflow::Type::SPOT_CHECK ) {

            uint32_t x = d;
            stats.spot_checked++;
            mpz_mul_ui(center, K, m);
            mpz_add_ui(next_p, center, x);
            auto s_start_t = high_resolution_clock::now();
            if (!mpz_probab_prime_p(next_p, 20)) {
                printf("\n\n");
                printf("%lu'th SPOT CHECK FAILED!\n", stats.spot_checked.load());
                printf("%lu * %u# / %u + %u is not prime!\n",
                        m, P, D, x);
                printf("\n\n");
                exit(1);
            }
            double total_s = duration<double>(high_resolution_clock::now() - s_start_t).count();
            stats.d_spot_check += total_s;
            continue;
        }

        if (m > max_m) max_m = m;

        assert( type == Overflow::Type::NEXT_PRIME );
        // stats.tested++ moved above.
        auto min_x = d;

        // This would mean that stop_x was poorly tuned.
        assert(min_x + 500 < ofs.coprime_X.back());

        { // Sieve and push that to overflow batch
            uint32_t min_x_i = ofs.coprime_index_next[min_x];
            uint32_t sieve_start = ofs.coprime_X[min_x_i];
            assert( min_x <= sieve_start);

            sieve_interval_cpu(
                m, true, sieve_start, ofs.coprime_X.back() - sieve_start + 1,
                composite_tmp, stats, tmp, K);

            push_to_overflow_batch(
                overflow_batch,
                m, min_x_i, sieve_start,
                K, center,
                composite_tmp, stats);

            if (overflow_batch.added == overflow_batch.N) {
                for (size_t i = 0; i < 30; i++) {
                    auto finished = run_overflow_batch(
                            runner,
                            overflow_batch,
                            MIN_GAP_TO_CONTINUE, MIN_MERIT,
                            K_log, P, D,
                            K, center, next_p, prev_p, tmp, tmp2,
                            stats, record_stream);
                    if (finished) break;
                    printf("GPUBatch didn't fully process any numbers?\n");
                }
            }
        }
    }

    if (is_running) {
        assert( (signed) overflow.size < og_config.cpu_threads ); // should contain only STOP_WORKER items

        if (overflow_batch.added) {
            // Clear out any remaining items queued in GPUBatch
            if (og_config.verbose >= 1) {
                printf("\tCPU overflow finishing %lu remaining items in GPUBatch\n", overflow_batch.added);
            }

            // Slightly akward to run partial batches so handle on CPU.
            GPUBatch &gpu_batch = overflow_batch.gpu_batch;
            for (uint32_t i = 0; i < overflow_batch.N; i++) {
                if (!gpu_batch.active[i])
                    continue;

                { // Run each remaining row of GPUBatch manually
                    auto& [m, x_i, sieve_start, next_gap] = overflow_batch.data[i];
                    if (next_gap == 0) {
                        uint32_t next_x = ofs.coprime_X[x_i];
                        uint32_t min_x = next_x;
                        run_tests_on_cpu(
                            MIN_GAP_TO_CONTINUE, MIN_MERIT,
                            K_log, P, D,
                            m, min_x,
                            K, center, next_p, prev_p, tmp, tmp2, composite_tmp, stats, record_stream);
                    } else {
                        // Ignores partially computed prev_prime.
                        mpz_mul_ui(center, K, m);
                        handle_next_prime_result(
                            MIN_GAP_TO_CONTINUE, MIN_MERIT,
                            K_log, P, D,
                            K, center,
                            m, next_gap,
                            next_p, prev_p, tmp, tmp2, stats, record_stream);
                    }
                }
                overflow_batch.remove_entry(i);
            }
        }
    }

    overflow.lock();
    stats.max_m = std::max(stats.max_m.load(), max_m);
    overflow.unlock();

    mpz_clears(K, center, next_p, prev_p, tmp, tmp2, NULL);
    if (og_config.verbose >= 3) {
        usleep(thread_index * 10'000); // X0ms
        printf("\tCPU overflow(%u) done\n", thread_index);
    }

}



uint32_t USE_GPU_FOR_OVERFLOW = true;

void run_overflow_coordinator_thread(const struct Config og_config) {
    try {
        {
            pthread_setname_np(pthread_self(), "CPU_OVERFLOW");
            std::ignore = nice(+1); // Lower priority a tiny bit
                                    // Helpers run at much lower
        }

        setup_overflow(og_config);

        TestingStats stats;

        std::vector<std::thread> worker_threads;
        for (int i = 0; i < og_config.cpu_threads; i++) {
            worker_threads.emplace_back(
                run_cpu_overflow_worker,
                i, std::ref(og_config), std::ref(stats)
            );
        }

        for (auto &worker : worker_threads) {
            worker.join();
        }

        if (is_running)
            assert( overflow.size == 0 ); // Should be empty now

        // How to get access to final m?
        uint64_t processed_m = (stats.max_m <= og_config.m_start) ?
            0 : count_num_m(og_config.m_start, stats.max_m - og_config.m_start, og_config.d);
        printf("Processed M: %'lu [%lu, %lu]\n",
                processed_m, og_config.m_start, stats.max_m.load());

        uint64_t T = stats.tested;
        if (og_config.verbose >= 1 and T > 0) {
            printf("\nCPU OVERFLOW Timing:\n");
            printf("\ttotal tested   : %lu (%.2f%% -> %.2f%% of total M)\n",
                    T,
                    100.0 * T / processed_m,
                    100.0 * stats.tested_prev / processed_m);
            printf("\t               :   (%.1f%% CPU, %.1f%% GPU)\n",
                    100.0 * stats.tested_cpu / T,
                    100.0 * stats.tested_gpu / T);
            printf("\t               :   (%lu CPU, %lu GPU)\n",
                    stats.tested_cpu.load(), stats.tested_gpu.load());
            printf("\tspot checked   : %lu (%.6f secs/prob_prime test)\n",
                    stats.spot_checked.load(), stats.d_spot_check / stats.spot_checked);
            printf("\tnext prime only: %lu, both sides: %lu\n",
                    stats.skipped_prev.load(), stats.tested_prev.load());
            printf("\ttotal time     : sieve: %.1f, cpu: %.1f, gpu %.1f\n",
                    stats.d_sieve.load(),
                    stats.d_next_prime_cpu.load(),
                    stats.d_next_prime_gpu.load());
            printf("\t               : prev_prime (cpu): %.1f, gpu misc: %.1f\n",
                    stats.d_prev_prime_cpu.load(),
                    stats.d_next_prime_gpu_misc.load());

            printf("\tnext prime test/sec sieve: %0.f, cpu: %.0f, gpu: %.0f\n",
                    (T + stats.tested_prev) / stats.d_sieve,
                    stats.tested_cpu / stats.d_next_prime_cpu,
                    stats.tested_gpu / stats.d_next_prime_gpu);
            uint32_t large = stats.greater_than_min_merit;
            if (large) {
                printf("\t> %.1f merit: %u\n", og_config.min_merit, large);
                if (stats.pseudoprimes || stats.mismatches) {
                    printf("\tMismatches | Fermat: %lu, Other: %lu\n",
                            stats.pseudoprimes.load(), stats.mismatches.load());
                }
            }
            int32_t missing = T - stats.tested_cpu - stats.tested_gpu;
            if (missing > 0) {
                printf("\tCPU+GPU tests don't add up %lu != %lu + %lu, missing %d\n",
                        T, stats.tested_cpu.load(),
                        stats.tested_gpu.load(), missing);
            }
            missing = T - stats.skipped_prev - stats.tested_prev;
            if (missing > 0) {
                printf("\tPrev tests don't add up %lu != %lu + %lu, missing %d\n",
                        T, stats.skipped_prev.load(),
                        stats.tested_prev.load(), missing);
            }
            printf("\n");
        }

    } catch (const std::exception &e) {
        cout << "ERROR in run_cpu_overflow_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}
