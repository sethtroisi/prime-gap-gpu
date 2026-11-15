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

/**
 * Benchmark GPUSieve and verify it bit-for-bit against a simple CPU sieve.
 *
 *   ./gpu_sieve_benchmark -p 353 -d 2310 --minc 50000000 --max-prime 1000
 *
 * Options not passed on to Args::argparse:
 *   --sieves <n>               X values per m range (default 40)
 *   --ranges <n>               m ranges (default 2)
 *   --verify <n>               sieves per range checked against the CPU sieve (default 2)
 *   --batch <n>                GPUSieve::set_batch_size(n)
 *   --max-pi <n>               max_p_i of GPUSieve::run
 *   --first-x <n>              skip X below n
 *   --finalize <none|cpu|gpu>  also time composite -> unknown m, on the CPU or with run_unknowns()
 *   --active-fraction <f>      fraction of m coprime to d that are still active (default 0.25)
 *   --profile                  GPUSieve::set_profile(true)
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gmp.h>
#include <primesieve.hpp>

#include "gap_common.h"
#include "gpu_sieve.h"

using std::vector;
using namespace std::chrono;


/** FNV-1a over the first `bits` bits. */
static uint64_t hash_bits(const uint64_t *data, uint64_t bits, uint64_t *popcount) {
    uint64_t h = 0xcbf29ce484222325ull;
    uint64_t count = 0;
    uint64_t words = bits / 64;
    for (uint64_t i = 0; i < words; i++) {
        h = (h ^ data[i]) * 0x100000001b3ull;
        count += __builtin_popcountll(data[i]);
    }
    if (bits % 64) {
        uint64_t last = data[words] & ((1ull << (bits % 64)) - 1);
        h = (h ^ last) * 0x100000001b3ull;
        count += __builtin_popcountll(last);
    }
    *popcount = count;
    return h;
}


static int64_t invert_mod(int64_t a, int64_t p) {
    int64_t x = 1, x1 = 0, a1 = a, p1 = p;
    while (p1) {
        int64_t q = a1 / p1;
        int64_t t = x1; x1 = x - q * x1; x = t;
        t = p1; p1 = a1 - q * p1; a1 = t;
    }
    return x < 0 ? x + p : x;
}


/** Simple, slow, independent CPU sieve. */
class CpuReference {
    public:
        CpuReference(const struct Config &config) : config(config) {
            mpz_t K;
            init_K(config, K);

            primesieve::iterator iter(3);
            for (uint64_t prime = iter.next_prime(); prime <= config.max_prime; prime = iter.next_prime()) {
                if (prime <= config.p && (config.d % prime > 0))
                    continue;

                const uint64_t base_r = mpz_fdiv_ui(K, prime);
                const uint64_t inv_K = invert_mod(base_r, prime);
                d_primes += prime <= config.p;
                primes.push_back(prime);
                neg_inv_Ks.push_back(prime - inv_K);
            }
            mpz_clear(K);
        }

        /** composite[i] = 1 if (m_start + 2*i + 1) * K + X has a prime factor in the first max_p_i primes. */
        void sieve(uint64_t m_start, uint64_t X, uint32_t max_p_i, vector<uint8_t> &composite) const {
            // max_p_i doesn't count the primes dividing d.
            const size_t count = std::min<size_t>(primes.size(), (size_t) d_primes + max_p_i);
            const uint64_t half = config.m_inc / 2;
            composite.assign(half, 0);
            uint8_t *c = composite.data();

            #pragma omp parallel for schedule(static, 1 << 16)
            for (size_t i = 0; i < count; i++) {
                const uint64_t p = primes[i];
                const uint64_t m_0 = ((X % p) * neg_inv_Ks[i]) % p;
                uint64_t mi_0 = (m_0 + p - (m_start % p)) % p;
                // Only odd mi are stored.
                if ((mi_0 & 1) == 0)
                    mi_0 += p;
                for (uint64_t t = mi_0 >> 1; t < half; t += p)
                    c[t] = 1; // benign race, all writers store 1
            }
        }

        size_t num_primes() const { return primes.size(); }

    private:
        const struct Config config;
        uint32_t d_primes = 0;
        vector<uint32_t> primes;
        vector<uint32_t> neg_inv_Ks;
};


int main(int argc, char *argv[]) {
    uint32_t sieves = 40;
    uint32_t ranges = 2;
    uint32_t verify = 2;
    bool profile = false;
    std::string finalize = "none";
    double active_fraction = 0.25;
    uint32_t batch = 0;
    uint32_t first_x = 0;
    uint32_t max_p_i = std::numeric_limits<uint32_t>::max();

    // Args gets everything else, after a default --mstart.
    vector<char*> args;
    static char default_mstart_flag[] = "--mstart";
    static char default_mstart[] = "80000000000";
    args.push_back(argv[0]);
    args.push_back(default_mstart_flag);
    args.push_back(default_mstart);
    for (int i = 1; i < argc; i++) {
        auto is = [&](const char* name) { return strcmp(argv[i], name) == 0 && i + 1 < argc; };
        if      (is("--sieves")) sieves = atoi(argv[++i]);
        else if (is("--ranges")) ranges = atoi(argv[++i]);
        else if (is("--verify")) verify = atoi(argv[++i]);
        else if (is("--batch"))  batch = atoi(argv[++i]);
        else if (is("--max-pi")) max_p_i = atoll(argv[++i]);
        else if (is("--first-x")) first_x = atoll(argv[++i]);
        else if (is("--finalize")) finalize = argv[++i];
        else if (is("--active-fraction")) active_fraction = atof(argv[++i]);
        else if (strcmp(argv[i], "--profile") == 0) profile = true;
        else args.push_back(argv[i]);
    }

    Config config = Args::argparse(args.size(), args.data(), Args::Pr::SEARCH_GPU);
    const bool finalize_ok = finalize == "none" || finalize == "cpu" || finalize == "gpu";
    if (config.valid == 0 || sieves == 0 || ranges == 0 || !finalize_ok) {
        Args::show_usage(argv[0], Args::Pr::SEARCH_GPU);
        printf("  [--sieves <n>] [--ranges <n>] [--verify <n>] [--batch <n>] [--max-pi <n>] [--first-x <n>] [--finalize none|cpu|gpu] [--active-fraction <f>] [--profile]\n");
        return 1;
    }
    if (config.m_inc % 2 || config.m_start % 2 || config.d % 2) {
        printf("m_start, m_inc and d must be even\n");
        return 1;
    }
    verify = std::min(verify, sieves);

    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
            printf("No CUDA device found\n");
            return 1;
        }
        printf("GPU             : %s\n", prop.name);
        printf("GPU SMs         : %d (compiled MP=%d)\n", prop.multiProcessorCount, MP);
    }
    printf("Config          : p=%u d=%u mstart=%lu minc=%lu max_prime=%lu\n",
            config.p, config.d, config.m_start, config.m_inc, config.max_prime);
    printf("Sieves          : %u x %u ranges (verify %u per range)\n", sieves, ranges, verify);
    if (max_p_i != std::numeric_limits<uint32_t>::max())
        printf("Max prime index : %u\n", max_p_i);

    vector<uint32_t> coprime_X = get_coprime_X(config, 10 * config.p);
    coprime_X.erase(coprime_X.begin(), std::lower_bound(coprime_X.begin(), coprime_X.end(), first_x));
    if (coprime_X.size() < sieves) {
        printf("Only %lu coprime X, lower --sieves\n", coprime_X.size());
        return 1;
    }

    const uint64_t BITS = config.m_inc / 2;

    std::unique_ptr<CpuReference> reference;
    if (verify) {
        auto t0 = high_resolution_clock::now();
        reference = std::make_unique<CpuReference>(config);
        printf("CPU reference   : %lu primes, setup %.1f seconds\n",
                reference->num_primes(), duration<double>(high_resolution_clock::now() - t0).count());
    }

    vector<double> sieve_ms;
    double gpu_seconds = 0;
    uint64_t checksum = 0;
    uint64_t total_composite = 0;
    uint64_t verified = 0;
    uint64_t mismatches = 0;
    uint64_t total_unknowns = 0;
    uint64_t unknown_mismatches = 0;

    // Like SieveData::active_m_i_bits (m coprime to d) but thinned to active_fraction.
    vector<uint64_t> active_bits((BITS + 63) / 64);
    auto setup_active_bits = [&](uint64_t m_start) {
        vector<uint8_t> coprime(BITS, 1);
        for (uint32_t p = 3; p <= config.d; p += 2) {
            if (config.d % p != 0) continue;
            for (uint64_t mi = 1; mi < config.m_inc; mi += 2) {
                if ((m_start + mi) % p == 0) {
                    for (uint64_t i = mi >> 1; i < BITS; i += p) coprime[i] = 0;
                    break;
                }
            }
        }
        std::fill(active_bits.begin(), active_bits.end(), 0);
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ m_start;
        const uint64_t threshold = active_fraction >= 1.0 ? ~0ull : (uint64_t) (active_fraction * 0x1p64);
        for (uint64_t i = 0; i < BITS; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            if (coprime[i] && rng <= threshold)
                active_bits[i >> 6] |= 1ull << (i & 63);
        }
    };
    // The finalize loop gap_search_primorial used before run_unknowns().
    auto cpu_finalize = [&](const uint64_t *composite, vector<uint32_t> &tests) {
        tests.clear();
        for (uint32_t i = 0; i < active_bits.size(); i++) {
            uint64_t active = active_bits[i];
            uint32_t partial = (64 * i) << 1;
            active &= ~composite[i];
            while (active != 0) {
                int r = __builtin_ctzl(active);
                uint64_t t = active & -active;
                active ^= t;

                uint32_t m_i = partial | (r<<1) | 1;
                tests.push_back(m_i);
            }
        }
    };
    vector<uint32_t> tests, expected_tests;

    {
        GPUSieve gpu_sieve(config);
        gpu_sieve.set_profile(profile);
        if (batch) gpu_sieve.set_batch_size(batch);
        printf("Batch           : %u\n", gpu_sieve.get_batch_size());

        vector<uint8_t> cpu_composite;

        // Untimed warm-up in a range that isn't benchmarked.
        gpu_sieve.run(config.m_start + ranges * config.m_inc, config.m_inc, coprime_X[0], max_p_i);
        for (uint32_t r = 0; r < ranges; r++) {
            const uint64_t m_start = config.m_start + r * config.m_inc;
            if (finalize != "none") setup_active_bits(m_start);
            for (uint32_t s = 0; s < sieves; s++) {
                const uint64_t X = coprime_X[s];

                auto t0 = high_resolution_clock::now();
                const uint64_t *gpu_composite = nullptr;
                if (finalize == "gpu") {
                    if (gpu_sieve.wants_active_bits(m_start, X, max_p_i))
                        gpu_sieve.set_active_bits(m_start, active_bits.data(), active_bits.size());
                    uint32_t count = 0;
                    const uint32_t *unknowns = gpu_sieve.run_unknowns(m_start, config.m_inc, X, max_p_i, &count);
                    tests.assign(unknowns, unknowns + count);
                } else {
                    gpu_composite = gpu_sieve.run(m_start, config.m_inc, X, max_p_i);
                    if (finalize == "cpu")
                        cpu_finalize(gpu_composite, tests);
                }
                auto t1 = high_resolution_clock::now();

                if (finalize == "gpu") {
                    // Not timed, same batch so this only copies composite.
                    gpu_composite = gpu_sieve.run(m_start, config.m_inc, X, max_p_i);
                    cpu_finalize(gpu_composite, expected_tests);
                    if (tests != expected_tests) {
                        if (unknown_mismatches++ < 5)
                            printf("\tUNKNOWNS MISMATCH m_start=%lu X=%lu | GPU: %lu, CPU: %lu\n",
                                    m_start, X, tests.size(), expected_tests.size());
                    }
                }
                total_unknowns += tests.size();
                const double ms = 1000 * duration<double>(t1 - t0).count();
                sieve_ms.push_back(ms);
                gpu_seconds += ms / 1000;

                uint64_t count;
                const uint64_t h = hash_bits(gpu_composite, BITS, &count);
                checksum = (checksum ^ h) * 0x100000001b3ull;
                total_composite += count;

                // First and last sieves of the range.
                const bool do_verify = (s == 0 && verify > 0) || (s + verify > sieves);
                if (do_verify) {
                    reference->sieve(m_start, X, max_p_i, cpu_composite);
                    uint64_t bad = 0;
                    for (uint64_t i = 0; i < BITS; i++) {
                        const uint8_t gpu_bit = (gpu_composite[i >> 6] >> (i & 63)) & 1;
                        if (gpu_bit != cpu_composite[i]) {
                            if (bad++ < 5)
                                printf("\tMISMATCH m=%lu (mi=%lu) X=%lu | CPU: %u, GPU: %u\n",
                                        m_start + 2 * i + 1, 2 * i + 1, X, cpu_composite[i], gpu_bit);
                        }
                    }
                    mismatches += bad;
                    verified += 1;
                }
                if (s < 3 || do_verify || config.verbose >= 2) {
                    printf("\tm_start=%lu X=%-4lu %8.3f ms | %lu/%lu composite | hash %016lx%s\n",
                            m_start, X, ms, count, BITS, h,
                            do_verify ? (mismatches ? " | VERIFY FAILED" : " | verified") : "");
                }
            }
        }
        // ~GPUSieve prints timing.
    }

    const double avg_ms = 1000 * gpu_seconds / sieve_ms.size();
    std::sort(sieve_ms.begin(), sieve_ms.end());

    printf("Avg sieve ms    : %.3f (min %.3f , median %.3f , max %.3f ) over %lu sieves\n",
            avg_ms, sieve_ms.front(), sieve_ms[sieve_ms.size() / 2], sieve_ms.back(),
            sieve_ms.size());
    printf("Sieves/second   : %.2f\n", 1000 / avg_ms);
    printf("Finalize        : %s", finalize.c_str());
    if (finalize != "none") {
        printf(" (active fraction %.2f , %.0f unknowns / sieve)", active_fraction, 1.0 * total_unknowns / sieve_ms.size());
    }
    printf("\n");
    mismatches += unknown_mismatches;
    printf("Composite       : %.4f%%\n", 100.0 * total_composite / (1.0 * BITS * sieves * ranges));
    printf("Checksum        : %016lx\n", checksum);
    if (verify || finalize == "gpu") {
        printf("Verify          : %s (%lu sieves, %lu mismatches)\n",
                mismatches ? "FAIL" : "PASS", verified, mismatches);
    } else {
        printf("Verify          : SKIPPED\n");
    }
    return mismatches ? 2 : 0;
}
