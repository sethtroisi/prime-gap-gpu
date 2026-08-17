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

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gmp.h>
#include "gap_common.h"

using std::cout;
using std::endl;
using std::vector;

class StatsCounters {
    public:
        StatsCounters(std::chrono::high_resolution_clock::time_point now) : s_start_t(now) {}

        const std::chrono::high_resolution_clock::time_point s_start_t;

        uint64_t  s_tests     = 0;

        /* PRP counter */
        uint64_t    s_total_prp_tests = 0;
        uint64_t    s_total_primes = 0;

        /* Number of times fallback or skips happened */
        uint64_t    s_skips_after_one_side = 0;
        uint64_t    s_gap_out_of_sieve_prev = 0;
        uint64_t    s_gap_out_of_sieve_next = 0;

        /* Interval stats */
        float     s_best_merit_interval = 0;
        uint64_t    s_best_merit_interval_m = 0;

        float     s_total_merit = 0.0;
        uint64_t    s_total_prev_p = 0;
        uint64_t    s_total_next_p = 0;

        // This can change in const possibly_print_stats
        mutable float     s_tests_per_second = 0;

        /** Return if stats were printed */
        void process_results(
            const Config &config,
            long m,
            int prev_p, int next_p,
            int p_tests, int n_tests,
            float merit);

        bool possibly_print_stats(
            const std::string name, const Config &config) const;
};


class GpuStatsCounters {
    public:
        GpuStatsCounters() {};

        uint64_t batches_run = 0;
        uint64_t batches_partial = 0;

        uint64_t wait_not_active = 0;

        double d_fill = 0.0;
        double d_queued_full = 0.0;
        double d_run = 0.0;
        double d_queued_done = 0.0;
        double d_results = 0.0;

        void reset() {
            batches_run = 0;
            batches_partial = 0;

            wait_not_active = 0;

            d_fill = 0.0;
            d_queued_full = 0.0;
            d_run = 0.0;
            d_queued_done = 0.0;
            d_results = 0.0;
        }

        void merge(GpuStatsCounters other) {
            batches_run     += other.batches_run;
            batches_partial += other.batches_partial;
            wait_not_active += other.wait_not_active;
            d_fill          += other.d_fill;
            d_queued_full   += other.d_queued_full;
            d_run           += other.d_run;
            d_queued_done   += other.d_queued_done;
            d_results       += other.d_results;
        }
};

