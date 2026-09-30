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

#include "gpu_testing.h"

#include <cassert>
#include <thread>
#include <tuple>

// pthread_setname_np
#include <pthread.h>

#include "gap_common.h"
#include "gap_stats.h"
#include "xoroshiro128plus.h"

// Comment out to use fake PRP test (for benchmarking)
#define GPU_TESTING

#ifdef GPU_TESTING
#include "miller_rabin.h"
#endif // GPU_TESTING


#ifdef GPU_TESTING

#ifdef GPU_BITS
const int BITS = GPU_BITS;
#else
const int BITS = 1024;
#endif

const int WINDOW_BITS = 4 + (BITS > 256) + (BITS > 512);
const int THREADS_PER_INSTANCE = (BITS <= 512) ? 4 : 8;

#endif // GPU_TESTING

/**
 * GPU_BATCHES the number of simultanious batches to create & queue.
 * GPU_BATCH_SIZE is 2^n | best is between 4K and 16K.
 */
const size_t GPU_BATCHES = 3;
const size_t GPU_BATCH_SIZE = 8 * 1024;



/********** BENCHMARKING ***********/
// Use `make BITS=X gpu_benchmark`
// 1080Ti 347# ???
// 1080Ti 151# ???
// 4070Ti Super 347# 43M PRP/second
// 4070Ti Super 151# 10.8M PRP/second
/********** BENCHMARKING ***********/

#ifdef GPU_TESTING
typedef mr_params_t<THREADS_PER_INSTANCE, BITS, WINDOW_BITS> gpu_params;
#endif // GPU_TESTING


class GPURunner::GPURunnerImpl {
    public:
        GPURunnerImpl() {}
        ~GPURunnerImpl() = default;

        void run(GPUBatch& batch) {
            #ifdef GPU_TESTING
                // run batch on gpu and wait for results to be set
                runner.run_test(batch.i, batch.z, batch.result);
            #else
                assert( batch.i % 16 == 0 );
                for (size_t gpu_i = 0; gpu_i < batch.i; gpu_i += 16) {
                    uint64_t rand = rng_next();
                    assert( gpu_i + 16 <= batch.i );
                    for (size_t j = 0; j < 16; j++) {
                        if (batch.active[gpu_i + j]) {
                            batch.result[gpu_i + j] = (rand & 3) == 0;
                        }
                        rand >>= 3;
                    }
                }
            #endif // GPU_TESTING
        }
    private:
#ifdef GPU_TESTING
        test_runner_t<gpu_params> runner{GPU_BATCH_SIZE};
#endif // GPU_TESTING
};

void GPURunner::run(GPUBatch& batch) {
    // Forward the call
    pImpl->run(batch);
}

GPURunner::GPURunner() : pImpl(std::make_unique<GPURunnerImpl>()) {}
GPURunner::~GPURunner() = default;


void gpu_state_and_checks(const mpz_t &K_in, const uint64_t m_end) {
    static_assert( GPU_BATCH_SIZE == 1024 || GPU_BATCH_SIZE == 2048 ||
                   GPU_BATCH_SIZE == 4096 || GPU_BATCH_SIZE == 8192 ||
                   GPU_BATCH_SIZE ==16384 || GPU_BATCH_SIZE ==32768 );

#ifdef GPU_TESTING
    printf("TESTING PRIMES ON GPU\n");
    printf("BITS=%d\n", BITS);
    printf("PRP/BATCH=%ld\n", GPU_BATCH_SIZE);
    printf("THREADS/PRP=%d\n", THREADS_PER_INSTANCE);
    printf("GPU_BATCHES=%lu\n", GPU_BATCHES);

    // +4 is just is personal safety blanket buffer.
    size_t N_bits = mpz_sizeinbase(K_in, 2) + log2(m_end) + 4;

    // P# roughly 349, 709, 1063, 1447
    for (size_t bits : {512, 1024, 1536, 2048, 3036, 4096}) {
        if (N_bits <= bits) {
            if (bits < BITS) {
                printf("\nFASTER WITH `make gap_search_gpu BITS=%ld` (may require `make clean`)\n\n", bits);
                exit(1);
            }
            break;
        }
    }
    if (N_bits >= BITS) {
        printf("\nERROR: GPU Compiled with BITS=%d but m*K has up to %lu bits\n\n",
                BITS, N_bits);
        exit(1);
    }
    assert( BITS <= (1 << (2 * WINDOW_BITS)) );
#else

    printf("FAKE PRIME TESTING\n");

#endif // GPU_TESTING
}
