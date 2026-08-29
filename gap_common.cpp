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
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <limits>

#include <gmp.h>

using std::cout;
using std::endl;
using std::pair;
using std::vector;
using namespace std::chrono;



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

vector<uint32_t> get_coprime_X(const struct Config& config, uint32_t max_x) {
    vector<uint32_t> X;

    vector<uint32_t> K_primes;
    for (auto prime : get_sieve_primes(config.p)) {
        if (config.d % prime != 0)
            K_primes.push_back(prime);
    }


    for (uint32_t x = 1; x <= max_x ; x += 1) {
        uint64_t any_coprime;
        any_coprime = false;

        for (auto prime : K_primes) {
            if (x % prime == 0) {
                any_coprime = true;
                break;
            }
        }

        if (!any_coprime) {
            if (config.d % 2 == 0 && x % 2 == 1) {
                // (m, d) = 1 -> m is odd
                // K is odd because no 2
                // m * K + X -> odd * odd + odd -> even
                continue;
            }
            X.push_back(x);
        }
    }

    return X;
}

void Args::show_usage(char* name, Pr program) {
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

    if (config.m_start >= 3 && config.m_start % 2 == 1) {
        cout << "Adjusting mstart to previous even." << endl;
        config.m_start -= 1;
    }

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
