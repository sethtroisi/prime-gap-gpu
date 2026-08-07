// Copyright 2020 Seth Troisi
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

#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmp.h>

using std::map;
using std::string;
using std::vector;

const double GAMMA = 0.577215665;

/* Arg Parsing */

struct Config {
    int valid   = 0;
    uint32_t p       = 0;
    uint32_t d       = 0;

    uint64_t m_start  = 0;
    uint64_t m_inc    = 0;
    // Only for testers, skip m < mskip
    uint64_t m_skip   = 0;

    float min_merit = 18;

    uint64_t max_prime    = 0;

    // GPU memory in MB allowed (probably ~10% error)
    int max_gpu_mem_mb = 4'000;

    /**
     * -1: results only
     *  0: results & final stats
     *  1: stats
     *  2: stats, probs, debug
     *  3: ???
     */
    int verbose = 2;

    // Show timing information (turn off --hide-timing for easier diff'ing)
    bool show_timing = true;

    // Secret option for testing code
    bool testing = false;
};


class Args
{
    public:
        enum Pr { SEARCH_GPU };

        static void show_usage(char* name, Pr program);
        static Config argparse(int argc, char* argv[], Pr program);
        static std::string gen_unknown_fn(const struct Config& config, std::string suffix);
        static int guess_compression(
            const struct Config& config,
            std::ifstream& unknown_file);

    private:
        // Disallow creating instance
        Args() = default;
};


class BitArrayHelper {
    public:
        BitArrayHelper(const struct Config& config, const mpz_t &K);

        /** Helper method for handling config.compression == 2 */
        vector<uint32_t> P_primes;
        vector<uint32_t> D_primes;

        /**
         * vector of x (in interval [-SL, SL]) with (K, x) == 1
         * values are storted [0, 2*SL] by adding +SL
         */
        vector<int32_t> coprime_X;

        /** is_offset_coprime[x] = ((K, x) == 1) */
        vector<char> is_offset_coprime;

        // When D % 2 == 0 =>  m % 2 == 1 => X % 2 == 0
        vector<int32_t> coprime_X_even;
        vector<char> is_offset_coprime_even;

        /**
         * (-K) % d, used to find first multiple of prime in: m * K + [-SL, SL]
         */
        uint32_t neg_K_mod_d;
        uint32_t SL_mod_d;
};

int64_t parse_unknown_line(
        const struct Config& config,
        const BitArrayHelper& helper,
        uint64_t m_expected,
        std::istream& input_line,
        vector<int32_t>& unknown_prev,
        vector<int32_t>& unknown_next);


/* Random Utils */

bool has_prev_prime_gmp();

uint64_t gcd(uint64_t a, uint64_t b);


/* K stuff */
double calc_log_K(const struct Config& config);

void init_K(const struct Config& config, mpz_t &K);

void K_stats(
        const struct Config& config,
        mpz_t &K, int *K_digits, double *K_log);


/* Utils */
size_t count_num_m(long ms, long mi, uint64_t d);

std::pair<vector<bool>, vector<uint32_t>>
    is_coprime_and_valid_m(const struct Config& config);

double prob_prime_and_stats(const struct Config& config, mpz_t &K);

/* Prime Stuff */

bool is_prime_brute(uint32_t n);
vector<uint32_t> get_sieve_primes(uint32_t n);

size_t primepi_estimate(uint64_t max_prime);
