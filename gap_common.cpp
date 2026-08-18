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

#include "gap_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <limits>

/* for dirname(3) */
#include <libgen.h>

/* for primesieve::iterator */
#include <primesieve.hpp>


using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::vector;
using namespace std::chrono;



static const std::map<uint64_t,uint64_t> common_primepi = {
    {       10'000'000,        664'579},
    {      100'000'000,      5'761'455},
    {      200'000'000,     11'078'937},
    {      400'000'000,     21'336'326},
    {      800'000'000,     41'146'179},
    {    1'000'000'000,     50'847'534},
    {    2'000'000'000,     98'222'287},
    {    3'000'000'000,    144'449'537},
    {    4'000'000'000,    189'961'812},
    {    5'000'000'000,    234'954'223},
    {    6'000'000'000,    279'545'368},
    {   10'000'000'000,    455'052'511},
    {   15'000'000'000,    670'180'516},
    {   20'000'000'000,    882'206'716},
    {   25'000'000'000,  1'091'987'405},
    {   30'000'000'000,  1'300'005'926},
    {   40'000'000'000,  1'711'955'433},
    {   50'000'000'000,  2'119'654'578},
    {   60'000'000'000,  2'524'038'155},
    {  100'000'000'000,  4'118'054'813},
    {  200'000'000'000,  8'007'105'059},
    {  300'000'000'000,  11'818'439'135},
    {  400'000'000'000,  15'581'005'657},
    {  500'000'000'000,  19'308'136'142},
    {1'000'000'000'000,  37'607'912'018},
    {2'000'000'000'000,  73'301'896'139},
    {3'000'000'000'000, 108'340'298'703},
    {4'000'000'000'000, 142'966'208'126},
    {5'000'000'000'000, 177'291'661'649}
};


uint64_t gcd(uint64_t a, uint64_t b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}


double _log(const mpz_t &K) {
    long exp;
    double mantis = mpz_get_d_2exp(&exp, K);
    return log(mantis) + log(2) * exp;
}


double calc_log_K(const struct Config& config) {
    mpz_t K;
    init_K(config, K);
    double log = _log(K);
    mpz_clear(K);
    return log;
}


void init_K(const struct Config& config, mpz_t &K) {
    mpz_init(K);
    mpz_primorial_ui(K, config.p);
    assert(0 == mpz_tdiv_q_ui(K, K, config.d));
    assert(mpz_cmp_ui(K, 1) > 0);  // K <= 1 ?!?
}


void K_stats(
        const struct Config& config,
        mpz_t &K, int *K_digits, double *K_log) {
    init_K(config, K);
    *K_log = _log(K);

    if (K_digits != nullptr) {
        int base10 = mpz_sizeinbase(K, 10);
        *K_digits = base10;

        if (config.verbose >= 1) {
        int K_bits   = mpz_sizeinbase(K, 2);
        printf("K = %d bits, %d digits, log(K) = %.2f\n",
                K_bits, base10, *K_log);
        }
    }
}


/**
 * Return estimated time (in seconds) to PRP test a composite with no small factor
 */
double prp_time_estimate_composite(double K_log, int verbose) {
    // TODO: For large K_log, time smaller PRP then upscale with polynomial

    // Some rough estimates at
    // https://github.com/sethtroisi/misc-scripts/tree/master/prime-time

    float K_log_2 = K_log * K_log;
    float t_estimate_poly = -1.1971e-03
        +  5.1072e-07 * K_log
        +  9.4362e-10 * K_log_2
        +  1.8757e-13 * K_log_2 * K_log
        + -1.9582e-18 * K_log_2 * K_log_2;
    float t_estimate = std::max(1e-3f, t_estimate_poly);

    // Not needed with GPU code.
    if (verbose >= 3) {
        if (t_estimate > 0.3) {
            printf("Estimated secs/PRP: %.1f\n", t_estimate);
        } else {
            // Benchmark in thread

            // Create some non-trivial semi-primes.
            mpz_t n, p, q;
            mpz_inits(n, p, q, nullptr);

            size_t bits = K_log * 1.442;
            assert( bits > 50 );

            mpz_set_ui(n, 1);

            // Multiply "large" static primes (25 bits+) to get number of size N
            size_t bit_goal = bits - 24;
            while (bit_goal > 0) {
                // Important to not ever choose small p
                size_t p_size = bit_goal < 50 ? bit_goal : 25;
                assert(p_size >= 25);
                mpz_ui_pow_ui(p, 2, p_size);
                mpz_nextprime(p, p);
                mpz_mul(n, n, p);
                bit_goal -= p_size;
            }
            mpz_set(p, n);

            // Smaller prime for fast nextprime.
            // Large enough to avoid being found with trial division.
            mpz_ui_pow_ui(q, 2, 25);
            mpz_nextprime(q, q);

            double t = 0;
            size_t count = 0;
            // time a reasonable number (or for 5 seconds)
            for (; count < 15 || t < 5; count++) {
                mpz_mul(n, p, q);
                assert( mpz_sizeinbase(n, 2) >= bits );

                auto  s_start_t = high_resolution_clock::now();

                assert( mpz_probab_prime_p(n, 25) == 0 );

                t += duration<double>(high_resolution_clock::now() - s_start_t).count();
                mpz_nextprime(q, q);
            }

            printf("Estimating PRP/s: %ld / %.2f = %.1f/s vs polyfit estimate of %.1f/s\n",
                    count, t, count / t, 1 / t_estimate);
            t_estimate = t / count;

            mpz_clears(n, p, q, nullptr);
        }
    }

    return t_estimate;
}


/**
 * Count of numbers coprime to d less than end; sum( gcd(m, d) == 1 for m in range(n, n+i) )
 * Uses inclusion exclusion on prime factorization of d
 */
static
uint64_t _r_count_num_m(uint64_t n, const vector<int> &factors_d, int i) {
    if (n == 0) return 0;
    if (i < 0) return n;

    return _r_count_num_m(n, factors_d, i-1) - _r_count_num_m(n / factors_d[i], factors_d, i-1);
}

/**
 * Count number of m [ms, ms + mi) coprime to d
 */
size_t count_num_m(long ms, long mi, uint64_t d) {
    if (d == 1)
        return mi;

    if (ms + mi < 10000) {
        size_t count = 0;
        for (long m = ms; m < ms + mi; m++)
            count += (gcd(m, d) == 1);
        return count;
    }

    vector<int> D_factors;
    {
        uint64_t temp = d;
        for (long p = 2; p*p <= temp; p++) {
            while (temp % p == 0) {
                D_factors.push_back(p);
                temp /= p;
            }
        }
        if (temp > 1)
            D_factors.push_back(temp);
    }

    return _r_count_num_m(ms + mi - 1, D_factors, D_factors.size()-1) -
           _r_count_num_m(ms - 1,      D_factors, D_factors.size()-1);
}


double prob_prime_and_stats(const struct Config& config, mpz_t &K) {
    int K_digits;
    double K_log;
    K_stats(config, K, &K_digits, &K_log);

    if (config.verbose >= 1) {
        // From Mertens' 3rd theorem
        double unknowns_after_sieve = 1 / (log(config.max_prime) * exp(GAMMA));
        const double N_log = K_log + log(config.m_start + config.m_inc / 2);
        const double prob_prime = 1 / N_log - 1 / (N_log * N_log);
        double prob_prime_after_sieve = prob_prime / unknowns_after_sieve;

        double unknowns_minus_K = 1;
        {
            uint64_t temp = config.d;
            for (long p = 2; p*p <= temp; p++) {
                if (temp % p == 0) {
                    unknowns_minus_K *= (p - 1.0) / p;
                    temp /= p;
                }
            }
            if (temp > 1)
                unknowns_minus_K *= (temp - 1.0) / temp;
        }
        // I think this is 2x off because of how primorials line up the residues
        // residuals aren't random like normal.
        unknowns_minus_K *= log(config.p) / log(config.max_prime);

        printf("\n");
        printf("\t%.3f%% of %d digit numbers are prime\n",
                100 * prob_prime, K_digits);
        printf("\tAPPROX %.1f%% of numbers tested = %.1f%% of X tested (after sieve to %luM)\n",
                100 * unknowns_after_sieve,
                100 * unknowns_minus_K,
                config.max_prime / 1000000);
        printf("\t%.3f%% of tests should be prime (%.1fx speedup)\n",
                100 * prob_prime_after_sieve, 1 / unknowns_after_sieve);
        printf("\t~ %.1f PRP tests per m (per side)\n",
                1 / prob_prime_after_sieve);
        printf("\n");
    }

    return K_log;
}


// Small sieve of Eratosthenes.
vector<uint32_t> get_sieve_primes(uint32_t n) {
    assert(n < 1'001'000); // Use libprimesieve for larger intervals

    vector<uint32_t> primes = {2};
    uint32_t half_n = n >> 1;
    vector<bool> is_prime(half_n + 1, true);

    for (uint32_t p = 3; p <= n; p += 2) {
        if (is_prime[p >> 1]) {
            primes.push_back(p);
            uint64_t p2 = p * p;
            if (p2 > n) break;

            for (uint32_t m = p2 >> 1; m <= half_n; m += p)
                is_prime[m] = false;
        }
    }
    for (uint32_t p = primes.back() + 2; p <= n; p += 2) {
        if (is_prime[p >> 1])
            primes.push_back(p);
    }
    return primes;
}


bool is_prime_brute(uint32_t n) {
    if ((n & 1) == 0)
        return false;
    for (uint32_t p = 3; p * p <= n; p += 2)
        if (n % p == 0)
            return false;
    return true;
}


size_t primepi_estimate(uint64_t max_prime) {
    // Lookup primepi for common max_prime values.
    if (common_primepi.count(max_prime)) {
        return common_primepi.at(max_prime);
    }
    return 1.04 * max_prime / log(max_prime);

}



void Args::show_usage(char* name, Pr program) {
    Config defaults;

    cout << "Usage: " << name << endl;
    cout << "[REQUIRED]" << endl;
    cout << "  -p <p>" << endl;
    cout << "  -d <p>" << endl;
    cout << "  --mstart <start>" << endl;
    cout << "  --minc   <int>" << endl;
    cout << endl;
    cout << "[OPTIONALLY]" << endl;
    cout << "  --min-merit <min_merit>" << endl;
    cout << "    only display prime gaps with merit >= min_merit" << endl;
if (program == Pr::SEARCH_GPU) {
    cout << "    allows for partial resume of a previous range" << endl;
}
if (program == Pr::SEARCH_GPU) {
    cout << "  --max-prime" << endl;
    cout << "    use primes <= max-prime (in millions) for checking composite" << endl;
    cout << endl;
    cout << "  --cpu-fraction <fraction of results to finalize on CPU>" << endl;
    cout << "  --cpu-threads <number of CPU threads for finalizing>" << endl;
}
    cout << endl;
    cout << "[OPTIONAL]" << endl;
    cout << "  -q, --quiet" << endl;
    cout << "    suppress some status output (twice for more suppression)" << endl;
    cout << "  -v, --verbose" << endl;
    cout << "    increase amount of output (twice for more verbosity)" << endl;
    cout << "  -h, --help" << endl;
    cout << "    print this help message" << endl;
}


Config Args::argparse(int argc, char* argv[], Pr program) {
    // NOTE: Remember to add to getopt_long(argc, argv, OPTIONS_STRING, ...) below
    static struct option long_options[] = {
        {"p",                required_argument, 0,  'p' },
        {"d",                required_argument, 0,  'd' },

        {"mstart",           required_argument, 0,   1  },
        {"minc",             required_argument, 0,   2  },

        {"min-merit",        required_argument, 0,   4  },

        {"max-prime",        required_argument, 0,   5  },

        {"cpu-fraction",     required_argument, 0,   7  },
        {"cpu-threads",      required_argument, 0,   8  },

        // Secret option
        {"hide-timing",      no_argument,       0,  11  },
        {"testing",          no_argument,       0,  12  },

        {"quiet",            no_argument,       0,  'q' },
        {"verbose",          no_argument,       0,  'v' },
        {"help",             no_argument,       0,  'h' },
        {0,                  0,                 0,   0  }
    };

    Config config;
    config.valid = 1;

    int option_index = 0;
    char c;
    while ((c = getopt_long(argc, argv, "qvhp:d:u:t:", long_options, &option_index)) >= 0) {
        switch (c) {
            case 'h':
                show_usage(argv[0], program);
                exit(0);

            case 'q':
                config.verbose--;
                break;

            case 'v':
                config.verbose++;
                break;

            case 'p':
                config.p = atoi(optarg);
                break;
            case 'd':
                config.d = atoi(optarg);
                break;

            case 1:
                config.m_start = atoll(optarg);
                break;
            case 2:
                config.m_inc = atoll(optarg);
                break;

            case 4:
                config.min_merit = atof(optarg);
                break;

            case 5:
                config.max_prime = atol(optarg) * 1'000'000;
                break;

            case 7:
                config.cpu_fraction = atof(optarg);
                break;
            case 8:
                config.cpu_threads = atoi(optarg);
                break;

            case 11:
                config.show_timing = false;
                break;

            case 12:
                config.testing = true;
                break;


            case 0:
                printf("option %s arg %s\n", long_options[option_index].name, optarg);
                config.valid = 0;
                break;
            case '?':
                config.valid = 0;
                break;
            default:
                config.valid = 0;
                printf("getopt returned \"%d\"\n", c);
        }
    }

    if (optind < argc) {
        config.valid = 0;
        printf("unknown positional argument: ");
        while (optind < argc) {
            printf("%s ", argv[optind++]);
        }
        printf("\n");
    }

    // ----- Validation

    if (config.m_start <= 0) {
        config.valid = 0;
        cout << "mstart must be greater than 0: " << config.m_start << endl;
    }

    int64_t last_m = config.m_start + 1000 * config.m_inc;
    if (last_m <= 0 || last_m > 10'000'000'000'001 ) {
        config.valid = 0;
        cout << "mstart + 1000 * minc must be <= 1e13" << endl;
    }

    if (config.m_inc <= 0) {
        config.valid = 0;
        cout << "m_inc must be greater than 0: " << config.m_inc << endl;
    }
    if (config.m_inc >= std::numeric_limits<int32_t>::max()) {
        config.valid = 0;
        cout << "m_inc must be less than 2B " << config.m_inc << endl;
    }

    if (config.max_prime < 1'000) {
        config.valid = 0;
        cout << "max_prime must be set" << endl;
    }
    if (config.max_prime > 2'000'000'000) {
        config.valid = 0;
        cout << "max_prime > 2B not supported" << endl;
    }

    if (0) { // Not needed anymore?
        uint64_t max_m = std::numeric_limits<int64_t>::max() / config.max_prime;
        if (max_m < 1000 || max_m <= (size_t) (last_m + 1000)) {
            config.valid = 0;
            printf("max_prime * last_m(%ld) would overflow int64, log2(...) = %.3f\n",
                last_m, log2(1.0 * last_m * config.max_prime));
        }
    }

    {
        // check if p is valid
        bool valid = config.p < 1'000'0000;
        for (size_t t = 2; valid && t*t <= config.p; t++) {
            valid = (config.p % t) > 0;
        }

        if (!valid) {
            config.valid = 0;
            cout << "p# not prime (p=" << config.p << ")" << endl;
            exit(1);
        }
    }

    if (config.d <= 0) {
        config.valid = 0;
        cout << "d must be greater than 0: " << config.d << endl;
    }

    if (config.cpu_fraction < .00001 || config.cpu_fraction > .1) {
        config.valid = 0;
        cout << "cpu-fraction must be between .00001 and .1: " << config.cpu_fraction << endl;
    }

    if (config.cpu_threads < 1 || config.cpu_threads > 159) {
        config.valid = 0;
        cout << "cpu-threads must be between 1 and 159: " << config.cpu_threads << endl;
    }

    if (config.valid == 0) {
        cout << endl;
    }

    return config;
}


BitArrayHelper::BitArrayHelper(const struct Config& config, const mpz_t &K) {
    const unsigned int D = config.d;

    neg_K_mod_d = mpz_cdiv_ui(K, D);
    if (D > 1) {
        assert(neg_K_mod_d != 0);
    }

    assert(D_primes.size() <= 9);  // 23# > 2^32
    assert( (config.d == 1) || (!D_primes.empty() && D_primes.front() >= 2) );
};
