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

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>


#include <gmp.h>

#include "gap_common.h"
#include "gap_stats.h"
#include "miller_rabin.h"


using namespace std::chrono;

//*************************************************************************** //
//**********************************GLOBALS********************************** //

// Can't import gpu_testing because it needs to import gap_seach_gpu.h / GPUBatch

#ifdef GPU_BITS
const int BITS = GPU_BITS;
#else
const int BITS = 1024;
#endif

const int WINDOW_BITS = (BITS <= 1024) ? 5 : 6;
const int THREADS_PER_INSTANCE = (BITS <= 512) ? 4 : 8;
const int ROUNDS = 1;
const size_t GPU_BATCHES = 3;
const size_t GPU_BATCH_SIZE = 8 * 1024;


std::atomic<bool> is_running{false};
std::mutex merge_mtx;
GpuStatsCounters global_stats;

//**********************************GLOBALS********************************** //
//*************************************************************************** //


void signal_callback_handler(int) {
    if (is_running == 1) {
       cout << endl;
       cout << "Caught CTRL+C stopping" << endl;
       is_running = 0;
    } else {
       cout << "Caught 2nd CTRL+C, exit(2) now." << endl;
       exit(2);
       exit(2);
    }
}


static
void run_benchmark_thread(const struct Config og_config) {
    const uint32_t N = GPU_BATCH_SIZE;

    typedef mr_params_t<THREADS_PER_INSTANCE, BITS, WINDOW_BITS> params;
    test_runner_t<params> runner(N, ROUNDS);

    GpuStatsCounters stats;

    mpz_t K, center, tmp;
    mpz_init(center);
    mpz_init(tmp);
    init_K(og_config, K);

    // Partial copy of GPUBatch
    vector<int>  result(N, 0);
    vector<mpz_t*> z;
    // XXX: This is an ugly hack because you can't create mpz_t vector easily
    mpz_t *z_array;
    z_array = (mpz_t *) malloc(N * sizeof(mpz_t));
    for (size_t i = 0; i < N; i++) {
        mpz_init(z_array[i]);
        z.push_back(&z_array[i]);

        // m * K + 12;
        mpz_mul_ui(center, K, og_config.m_start + 2 * i + 1);
        mpz_add_ui(*z[i], center, 12);
    }

    while (is_running) {
        auto t0 = high_resolution_clock::now();
        runner.run_test(N, z, result);
        auto t1 = high_resolution_clock::now();
        stats.total_prp_tests += N;
        stats.total_primes += std::count(result.begin(), result.end(), 1);
        stats.batches_run += 1;

        auto t2 = high_resolution_clock::now();
        double d1 = duration<double>(t1 - t0).count();
        double d2 = duration<double>(t2 - t1).count();
        stats.d_run += d1;
        stats.d_results += d2;
    }

    {
        merge_mtx.lock();
        global_stats.merge(stats);
        merge_mtx.unlock();
    }

    mpz_clear(center);
    mpz_clear(tmp);
    mpz_clear(K);
}




int main(int argc, char* argv[]) {
    Config config = Args::argparse(argc, argv, Args::Pr::SEARCH_GPU);

    if (config.valid == 0) {
        Args::show_usage(argv[0], Args::Pr::SEARCH_GPU);
        return 1;
    }

    printf("TESTING PRIMES ON GPU\n");
    printf("BITS=%d\n", BITS);
    printf("PRP/BATCH=%ld\n", GPU_BATCH_SIZE);
    printf("THREADS/PRP=%d\n", THREADS_PER_INSTANCE);
    printf("GPU_BATCHES=%lu\n", GPU_BATCHES);

    is_running = true;

    // Setup CTRL+C catcher
    signal(SIGINT, signal_callback_handler);

    auto start_t = high_resolution_clock::now();

    std::thread threads[GPU_BATCHES];
    for(size_t i = 0; i < GPU_BATCHES; i++) {
        threads[i] = std::thread(
                run_benchmark_thread,
                std::ref(config)
        );
    }

    while (is_running) {
        usleep(10'000); // 10ms;
    }

    for (auto &thread : threads) {
        thread.join();
    }

    double total_t = duration<double>(high_resolution_clock::now() - start_t).count();
    setlocale(LC_NUMERIC, "");
    printf("\nGPU Timings (%.0f seconds):\n", total_t);
    printf("\ttotal tests     : %'lu (%.1f%% prime) (%'u/sec)\n",
            global_stats.total_prp_tests,
            100.0 * global_stats.total_primes / global_stats.total_prp_tests,
            (uint32_t) (global_stats.total_prp_tests / total_t));
    printf("\ttotal batches   : %'lu (%.5f secs/batch)\n",
            global_stats.batches_run, total_t / global_stats.batches_run);
    printf("\tfilling batches : %.1f seconds (%.1f%%)\n",
            global_stats.d_fill, 100 * global_stats.d_fill / total_t);
    printf("\trunning on gpu  : %.1f seconds (%.1f%%)\n",
            global_stats.d_run, 100 * global_stats.d_run / total_t);
    printf("\tresults         : %.1f seconds (%.1f%%)\n",
            global_stats.d_results, 100 * global_stats.d_results / total_t);
    printf("\n");
    setlocale(LC_NUMERIC, "C");
}
