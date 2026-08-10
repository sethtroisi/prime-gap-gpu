// Copyright 2021 Seth Troisi
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

#include "gap_test_common.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <sstream>

#include "gap_common.h"

using std::cout;
using std::endl;
using std::vector;


void StatsCounters::process_results(
        const Config &config,
        long m,
        int prev_p, int next_p,
        int p_tests, int n_tests,
        float merit) {

    s_tests += 1;

    // TODO break out s_p_tests, s_n_tests;
    s_total_prp_tests += p_tests + n_tests;

    // TODO s_side_skips

    s_total_merit += merit;
    s_total_prev_p += std::max(0, prev_p);
    s_total_next_p += std::max(0, next_p);

    if (merit > s_best_merit_interval) {
        s_best_merit_interval = merit;
        s_best_merit_interval_m = m;
    }

    if (possibly_print_stats("CPU", config)) {
        s_best_merit_interval = 0;
        s_best_merit_interval_m = 0;
    }
}

bool StatsCounters::possibly_print_stats(
        const std::string name,
        const Config &config) const {

    // truncate to a nearby multiple of 10000 (avoid making zero)
    size_t print_interval = 1800 * s_tests_per_second;
    if (print_interval > 10000)
        print_interval -= (print_interval % 10000);

    // if s_tests = {1,3,5} * 10 ^ x
    bool is_power_print = (s_tests == 1);
    for (uint64_t p = 100; p <= s_tests; p *= 10) {
        is_power_print |= (s_tests == p) || (s_tests == 3*p) || (s_tests == 5*p);
    }

    if ( is_power_print || (print_interval > 0 && s_tests % print_interval == 0) ) {
        auto s_stop_t = std::chrono::high_resolution_clock::now();
        double   secs = std::chrono::duration<double>(s_stop_t - s_start_t).count();
        s_tests_per_second = s_tests / secs;

        if (config.verbose >= 1) {
            // Stats!
            if (s_tests > secs) {
                printf("\t%s tests     %-10lu (%.2f/sec)  %.0f seconds elapsed\n",
                    name.c_str(), s_tests, s_tests / secs, secs);
            } else {
                printf("\t%s tests     %-10lu (%.2f secs/test)  %.0f seconds elapsed\n",
                    name.c_str(), s_tests, secs / s_tests, secs);
            }

            printf("\t    prp tests %-10ld (avg: %.2f) (%.1f tests/sec)\n",
                s_total_prp_tests,
                s_total_prp_tests / (float) s_tests,
                s_total_prp_tests / secs);

            if (config.verbose >= 2) {
                // Suppress for now, this is almost exactly 100 - extra sieves prev_gap %
                if (s_skips_after_one_side) {
                    printf("\t    only next_prime %ld (%.2f%%)\n",
                        s_skips_after_one_side, 100.0 * s_skips_after_one_side / s_tests);
                }
                if (s_gap_out_of_sieve_prev + s_gap_out_of_sieve_next > 0) {
                    printf("\t    extra sieves prev_gap %ld (%.2f%%), next_gap %ld (%.2f%%)\n",
                        s_gap_out_of_sieve_prev, 100.0 * s_gap_out_of_sieve_prev / s_tests,
                        s_gap_out_of_sieve_next, 100.0 * s_gap_out_of_sieve_next / s_tests);
                }

                printf("\t    best merit this interval: %.3f (at m=%ld)\n",
                    s_best_merit_interval, s_best_merit_interval_m);

                /*
                printf("\t    avg merit: %.3f, avg gap: %.1f (%lu + %lu)\n",
                    s_total_merit / s_tests,
                    ((float) s_total_prev_p + s_total_next_p) / s_tests,
                    s_total_prev_p / s_tests,
                    s_total_next_p / s_tests
                );
                */
            }
            return true;
        }
    }
    return false;
}
