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
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
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

    float min_merit = 18;

    uint64_t max_prime    = 0;

    // Fraction of results to handle on CPU.
    float cpu_fraction = 0.005;
    int cpu_threads = 3;

    /**
     * -1: results only
     *  0: results & final stats
     *  1: stats, probs,
     *  2: debug
     *  3: traces
     */
    int verbose = 1;

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

    private:
        // Disallow creating instance
        Args() = default;
};


/* Random Utils */

uint64_t gcd(uint64_t a, uint64_t b);


/* K stuff */
double _log(const mpz_t &K);
double calc_log_K(const struct Config& config);

void init_K(const struct Config& config, mpz_t &K);

void K_stats(
        const struct Config& config,
        mpz_t &K, int *K_digits, double *K_log);


/* Utils */
size_t count_num_m(long ms, long mi, uint64_t d);

double prob_prime_and_stats(const struct Config& config, mpz_t &K);

/** vector of X such that gcd(K, X[i]) == 1, 1 <= X[i] <= max_x */
vector<uint32_t> get_coprime_X(const struct Config& config, uint32_t max_x);


/* Prime Stuff */
vector<uint32_t> get_sieve_primes(uint32_t n);
