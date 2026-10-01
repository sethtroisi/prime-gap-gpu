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

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

// pthread_setname_np
#include <pthread.h>

#include <gmp.h>
#include <primesieve.hpp>

#include "gap_common.h"
#include "gap_stats.h"
#include "gap_linear_testing.h"
#include "gpu_testing.h"
#include "overflow.h"


//#define GPU_SIEVE

//#define CPU_VERIFY
#define CPU_SIEVE (!defined(GPU_SIEVE) || defined(CPU_VERIFY))

#ifdef GPU_SIEVE
#include "gpu_primorial_sieve.h"
#endif // GPU_SIEVE

const bool EXTRA_CHECKS = false;


using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using namespace std::chrono;


//*************************************************************************** //
//*********************************FORWARDS********************************** //

class SieveData;

//*********************************FORWARDS********************************** //
//*************************************************************************** //



//*************************************************************************** //
//**********************************GLOBALS********************************** //

/** Shared state between threads in gap_search_common */
// std::atomic<bool> is_running;
// std::atomic<uint8_t> stop_queue{0};

// Don't read from sieve_data without holding sieve_mtx
std::mutex sieve_mtx;
std::unique_ptr<SieveData> sieve_data;

const vector<uint16_t> VALID_GAPS = {
    10624, 10648, 10688, 10690, 10852, 10878,
    10930, 10946, 10954, 10972, 11002, 11038,
    11060, 11062, 11068, 11080, 11092, 11098,
    11104, 11114, 11116, 11122, 11138, 11146,
    // And everything bigger
};

//**********************************GLOBALS********************************** //
//*************************************************************************** //


//*************************************************************************** //
//*********************************CONSTANTS********************************** //

const size_t OPEN_SIEVES = 1;

//*********************************CONSTANTS********************************** //
//*************************************************************************** //


void prime_gap_test(const struct Config config);

int main(int argc, char* argv[]) {
    Config config = Args::argparse(argc, argv, search_type::SEARCH_LINEAR_GPU);

    if (config.valid == 0) {
        Args::show_usage(argv[0], search_type::SEARCH_LINEAR_GPU);
        return 1;
    }

    if (config.verbose >= 2) {
        printf("Compiled with GMP %d.%d.%d\n",
            __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR, __GNU_MP_VERSION_PATCHLEVEL);
    }

    if (config.max_prime < 2'000'000 || 1'500'000'000 < config.max_prime) {
        printf("\tmax_prime(%'ld) should be between 2M and 1.5B\n", config.max_prime);
        return 1;
    }

    if (config.m_inc < 1'000'000) {
        printf("\t--minc should be at least 1 million\n");
        return 1;
    }

    setlocale(LC_NUMERIC, "");
    if (config.verbose >= 0) {
        printf("\n");
        printf("Testing K=%d^%d\n", config.p, config.d);
    }
    setlocale(LC_NUMERIC, "C");

    prime_gap_test(config);

    if (config.verbose >= 2)
        cout << argv[0] << " ended" << endl;
}


class SieveData {
    public:
        SieveData(const struct Config config);

        /**
         * NEW -> ACTIVE -> FINAL -> DONE
         * FIRST_SIEVE => Running the first sieve
         * FINAL => Don't sieve any more, just finish outstanding prime tests.
         *      would be kinda nice to start on next sieves but IDK how to avoid that delay.
         */
        enum State { NEW, FIRST_SIEVE, ACTIVE, FINAL, DONE };
        State state = NEW;

        struct Config config;

        size_t current_offset = 0;
        size_t length = 0;

        std::atomic<uint8_t> sieves_ready{0};
        std::pair<uint32_t, vector<uint32_t>> next_sieve;

        /** sieve_mtx must be held while calling all methods*/
        void setup_sieve_data(bool stop_after);
        bool try_set_testing_data(LinearTestData &testing);
        void go_to_next_range();
};


SieveData::SieveData(const struct Config config) {
    this->config = config;

    sieves_ready = 0;
}

/** sieve_mtx must be held while calling */
void SieveData::setup_sieve_data(bool stop) {
    // Verify stuff
    assert(state == SieveData::NEW);

    current_offset = 0;
    length = 0;

    sieves_ready = 0;
    sieves_ready.notify_all();
    next_sieve.first = 0;
    next_sieve.second.clear();

    if (stop)
        return;

    if (config.verbose + (config.m_start <= 1) >= 3)
        printf("\nSetup, starting at offset=%lu length=%luM\n",
                current_offset, length / 1000000);

    state = SieveData::FIRST_SIEVE;
}

/** sieve_mtx, test_data.lock() must be held while calling */
bool SieveData::try_set_testing_data(LinearTestData &testing) {
    if (this->state == NEW) {
        return false;
    }

    assert( testing.state == LinearTestData::WAITING );
    assert( testing.active_batches == 0 );
    assert( testing.running_batches == 0 );
    // TODO TBD what this should be
    //assert( current_testing_x > 0 ); // 0 is the sentinal in next_sieve.

    // Look for finished sieve to copy over.
    //for (uint32_t i = 0; i < OPEN_SIEVES; i++) {
    {
        auto &[start, next_composites] = next_sieve; //[i];
        // TODO TBD how this works with the senital of 0
        if (start == current_offset) {
            // Set testing data.
            testing.offset = start;
            //testing.length = ???
            testing.composites.swap(next_composites);

            testing.state = LinearTestData::ACTIVE;
            testing.state.notify_one();

            next_composites.clear();

            sieves_ready--;
            sieves_ready.notify_all();
            assert(next_sieve.first == 0);
            assert(next_sieve.second.empty());
            return true;
        }
    }

    return false;
}

/** sieve_mtx must be held while calling */
void SieveData::go_to_next_range() {
    current_offset += length;
    if (config.verbose >= 3) {
        printf("\tMoving to offset=%ld\n", current_offset);
    }
}

static
void run_sieve_thread(std::atomic<uint8_t> &setup_done) {
    try {
        pthread_setname_np(pthread_self(), "SIEVE_THREAD");
        std::ignore = nice(-2); // Increase priority a bit

        std::unique_lock<std::mutex> lock(sieve_mtx, std::defer_lock);

        auto s_thread_start_t = high_resolution_clock::now();

        // Some prework
        mpz_t K;
        struct Config config = sieve_data->config;
        init_K(config, K);

        assert ( mpz_even_p(K) == true ); // Makes math below easier if true
        assert ( config.m_start % 2 == 0); // always start on even
        assert ( config.m_inc % 2 == 0); // all future m_start are even

        assert(sieve_data);
        const auto m_inc = config.m_inc;

        uint32_t all_primes_count = primesieve::count_primes(3, config.max_prime);
#if CPU_SIEVE
        vector<std::pair<uint32_t, uint32_t>> p_and_start_wheel;
        vector<std::pair<uint32_t, uint32_t>> p_and_start_small;
        vector<std::pair<uint32_t, uint32_t>> p_and_start_large;
        p_and_start_large.reserve(all_primes_count);
        {
            primesieve::iterator iter;
            uint64_t prime = iter.next_prime();
            assert (prime == 2);  // we skip 2 which is the oddest prime.
            for (prime = iter.next_prime(); prime < config.max_prime; prime = iter.next_prime()) {
                // First multiple of 2*prime after K.
                uint64_t base_r = mpz_fdiv_ui(K, 2*prime);
                // TODO record mod3 for small

                assert( base_r % 2 == 0 );
                base_r /= 2;
                // Can be 0 when prime divides p
                assert( 0 <= base_r && base_r < prime );

                if (prime <= 11) {
                    p_and_start_wheel.emplace_back(prime, base_r);
                } else if (prime < 100'000) {
                    p_and_start_small.emplace_back(prime, base_r);
                } else {
                    p_and_start_large.emplace_back(prime, base_r);
                }
            }
        }

        const auto M_INC_HALF = m_inc / 2;
        // Need to be able to write to composites[m_inc] as sentinel
        vector<uint64_t> composites(M_INC_HALF / 64 + 1, 0);
#endif  // CPU_SIEVE

#ifdef GPU_SIEVE
        GPUPrimorialSieve gpu_sieve(config);
#endif // GPU_SIEVE

        // ~10KB
        vector<uint64_t> wheel(64 * 3*5*7*11);

        uint64_t total_range = 0;
        uint64_t total_runs = 0;
        uint64_t total_unknown = 0;
        double total_time = 0;
        double finalize_time = 0;

        setup_done = 1;
        setup_done.notify_all();

        while (is_running && stop_queue <= 1) {
            lock.lock();
            uint64_t offset = sieve_data->current_offset;
            const auto state = sieve_data->state;
            if (state == SieveData::FIRST_SIEVE) {
                if (config.m_start != sieve_data->config.m_start) {
                    if (config.verbose >= 3)
                        printf("Reset GPU Sieve to 0\n");
                }
            }

            if ((state != SieveData::FIRST_SIEVE && state != SieveData::ACTIVE)
                    || offset == 0) {
                lock.unlock();
                usleep(1'000); // 1ms
                continue;
            }

            // Check if any empty next_sieves.
            {
                if (sieve_data->sieves_ready == OPEN_SIEVES) {
                    lock.unlock();
                    sieve_data->sieves_ready.wait(OPEN_SIEVES);
                    continue;
                }
            }

            config = sieve_data->config;

            lock.unlock();

            auto s_start_t = high_resolution_clock::now();
            const uint64_t m_start = config.m_start;

            assert(m_start % 2 == 0); // or fix the code

#if CPU_SIEVE
            // Don't need fill because wheel sets (not or's)
            // std::fill(composites.begin(), composites.end(), 0);

            { // Handle all divisors of d at one time.
                std::fill(wheel.begin(), wheel.end(), 0);
                uint32_t wheel_bits = 64 * wheel.size();

                for( auto& temp : p_and_start_wheel) {
                    const auto [p, start] = temp;
                    assert (p != 2);

                    // mark all later multiples
                    uint32_t i = start;
                    for( ; i < wheel_bits; i += p ) {
                        wheel[i >> 6] |= 1ull << (i & 63);
                    }

                    temp.second = (start + M_INC_HALF) % (2*p);
                }

                // wheel tiled
                for(uint32_t c_i = 0; c_i < composites.size(); ) {
                    size_t copy = std::min(composites.size() - c_i, wheel.size());
                    for (uint32_t j = 0; j < copy; j ++) {
                        composites[c_i + j] = wheel[j];
                    }
                    c_i += copy;
                }
            }

            // Break the larger range up into smaller ranges that are more likely to fit in L2 (2MB cache)
            uint64_t intervals = M_INC_HALF / 995'000 + 1;
            //printf("Breaking up into %lu intervals of %lu each\n", intervals, m_inc / intervals);
            for (size_t interval = 0; interval < intervals; interval++) {
                // indexed into [0, m_inc)
                uint64_t i_start = interval * M_INC_HALF / intervals;
                uint64_t i_end   = (interval+1) * M_INC_HALF / intervals;

                //printf("%2lu -> [%lu, %lu) = %lu\n", interval, i_start, i_end, interval_half_length);
                for( auto& temp : p_and_start_small) {
                    uint32_t prime = temp.first;
                    uint32_t t = temp.second;
                    assert( i_start <= t );
                    // TODO maybe try to avoid mults of 3
                    for (; t < i_end; t += prime) {
                        composites[t >> 6] |= 1ul << (t & 63);
                    }
                    temp.second = t;
                }
            }
            // Update all starts
            for( auto& temp : p_and_start_small) {
                assert( temp.second >= M_INC_HALF );
                temp.second -= M_INC_HALF;
            }

            for( auto& temp : p_and_start_large) {
                uint32_t prime = temp.first;
                uint32_t t = temp.second;
                for (; t < M_INC_HALF; t += prime) {
                    composites[t >> 6] |= 1ul << (t & 63);
                }
                temp.second = t - M_INC_HALF;
            }
#endif  // CPU_SIEVE

#ifdef GPU_SIEVE
            // TODO get exact prime counts to agree.
            uint64_t *gpu_composites = gpu_sieve.run(
                    m_start, m_inc, 0, prime_count);
#endif // GPU_SIEVE

#ifdef CPU_VERIFY
            {
                uint32_t num_cpu_composite = 0;
                for (auto c : composites) {
                    num_cpu_composite += std::popcount(c);
                }
                uint32_t num_gpu_composite = 0;
                for (uint32_t m_i = 0; m_i < (M_INC_HALF+7)/8; m_i += 1) {
                    num_gpu_composite += std::popcount(gpu_composites[m_i]);
                }

                uint32_t mismatches = 0;
                for (uint32_t m_i = 1; m_i < m_inc; m_i += 2) {
                    uint32_t t = m_i >> 1;
                    uint8_t cpu_bit = (    composites[t >> 6] & (1 << (t & 63))) > 0;
                    uint8_t gpu_bit = (gpu_composites[t >> 6] & (1 << (t & 63))) > 0;
                    bool mismatch = gpu_bit != cpu_bit;
                    mismatches += mismatch;
                    if (mismatch && mismatches < 10) {
                        printf("Mismatch at m=%lu (%u) | X=%lu | CPU: %u, GPU: %u\n",
                                m_start + m_i, m_i, X, cpu_bit, gpu_bit);
                    }
                }
                printf("GPU/CPU sieve mismatches: %u | composites CPU: %u GPU: %u\n",
                        mismatches, num_cpu_composite, num_gpu_composite);
                if (mismatches) {
                    is_running = false;
                    exit(0);
                }
            }
#endif  // CPU_VERIFY

            auto s_stop_t = high_resolution_clock::now();
            double sieve_duration_t = duration<double>(s_stop_t - s_start_t).count();
            total_runs += 1;
            total_time += sieve_duration_t;

            lock.lock();

            assert( offset != sieve_data->current_offset );

            double finalize_duration_t;
            uint64_t num_unknowns;
            { // Finalize
                auto s_start_t = high_resolution_clock::now();

                vector<uint32_t> *tests = nullptr;
                /*
                for (auto& t : sieve_data->next_sieves) {
                    if (t.first == 0) {
                        t.first = X;
                        tests = &t.second;
                        sieve_data->sieves_ready++;
                        sieve_data->sieves_ready.notify_all();
                        break;
                    }
                }*/
                auto &t = sieve_data->next_sieve;
                assert( t.first == 0 );
                t.first = offset;
                tests = &sieve_data->next_sieve.second;
                sieve_data->sieves_ready++;
                sieve_data->sieves_ready.notify_all();

                assert( tests != nullptr );
                assert( tests->empty() );

                /*
                const vector<uint64_t> &active_bits = sieve_data->get_active_bits();
                for (uint32_t i = 0; i < active_bits.size(); i++) {
                    uint64_t active = active_bits[i];
                    uint32_t partial = (64 * i) << 1;
#ifdef GPU_SIEVE
                    //active ^= (active & gpu_composites[i]);
                    active &= ~gpu_composites[i];
#else
                    active &= ~composites[i];
#endif  // GPU_SIEVE
                    // Find all set bits in active
                    while (active != 0) {
                        int r = __builtin_ctzl(active);
                        uint64_t t = active & -active;
                        active ^= t;

                        uint32_t m_i = partial | (r<<1) | 1;
                        tests->push_back(m_i);
                    }
                }
                */
                for (auto c : composites) {
                    num_unknowns += 32 - std::popcount(c);
                }

                // TODO record number of bits in composites.
                total_range += m_inc;
                total_unknown += num_unknowns;

                auto s_stop_t = high_resolution_clock::now();
                finalize_duration_t = duration<double>(s_stop_t - s_start_t).count();
                finalize_time += finalize_duration_t;
                total_time += finalize_duration_t;

                // Move to next range.
                sieve_data->current_offset += m_inc;
                if (state == SieveData::FIRST_SIEVE) {
                    // Mark as active after next_sieves is set.
                    sieve_data->state = SieveData::ACTIVE;
                }
            }

            if ((config.verbose + (config.m_start <= 1'000'000'000)) >= 2) {
                printf("\tSieve o=%lu with %lu/%lu (%.0f%%) unknown/active"
                       " took %.3f + %.3f seconds\n",
                       offset, num_unknowns, m_inc,
                       100.0 * num_unknowns / m_inc,
                       sieve_duration_t, finalize_duration_t);
            }

            lock.unlock();
        }

        mpz_clear(K);

        if (config.verbose >= 1 && total_runs > 0) {
            double total_s = duration<double>(high_resolution_clock::now() - s_thread_start_t).count();
            setlocale(LC_NUMERIC, "");
            printf("\nSIEVE Timings:\n");
            printf("\tsieves : %lu (%.3f/second, %.1f%% of total time)\n",
                    total_runs, total_runs / total_s, 100.0 * total_time / total_s);
            printf("\ttotal range: %'lu, unknown: %'lu (%.2f%%)\n",
                    total_range, total_unknown, 100.0 * total_unknown / total_range);
            printf("\tunknown / run: %'lu\n",
                    total_unknown / total_runs);
            printf("\t---------------------------------------\n");
            printf("\ttotal time            : %.1f seconds (%.4f/sieve, %.3f secs/billion)\n",
                    total_time, total_time / total_runs, total_time / total_runs * 1e9 / m_inc);
            printf("\tfinalize time (%4.1f%%) : %.1f seconds (%.4f/sieve)\n",
                    100 * finalize_time / total_time, finalize_time, finalize_time / total_runs);
            printf("\n");
            setlocale(LC_NUMERIC, "C");
        }
    } catch (const std::exception &e) {
        cout << "ERROR in run_sieve_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


static
void run_testing_thread(const struct Config og_config) {
    try {
        pthread_setname_np(pthread_self(), "TESTING_THREAD");
        std::ignore = nice(-2); // Increase priority a bit
        cout << endl;

        mpz_t K;
        init_K(og_config, K);

        // Print Header info
        if (og_config.verbose >= 1) {
            setlocale(LC_NUMERIC, "");
            printf("\nTesting ranges of %'ld\n", og_config.m_inc);
            setlocale(LC_NUMERIC, "C");
        }

        /* Note: Uses a double batched system
         * C++ Thread is preparing batch_a (even more m), while GPU runs batch_b */
        std::deque<GPUBatch> gpu_batches;
        for (uint32_t i = 0; i < GPU_BATCHES; i++) {
            gpu_batches.emplace_back(GPU_BATCH_SIZE);
        }
        LinearTestData test_data{og_config};

        std::thread gpu_threads[GPU_BATCHES];
        for(size_t i = 0; i < GPU_BATCHES; i++) {
            // Barely needs gpu_batches to be owned here.
            gpu_threads[i] = std::thread(run_gpu_thread,
                    i, og_config.verbose,
                    std::ref(test_data),
                    std::ref(gpu_batches[i]),
                    std::ref(K)
            );
        }

        // Main loop
        while (is_running && stop_queue <= 1) {
            /**
             * Try to fill all batches
             * queue on gpu all ready batches
             * wait for result from the 1st batch sent
             */

            const auto state = test_data.state.load();

            if (state == LinearTestData::WAITING) {
                sieve_mtx.lock();
                test_data.lock();

                [[maybe_unused]] uint8_t had_ready = sieve_data->sieves_ready;
                bool set = sieve_data->try_set_testing_data(test_data);
                if (!set) test_data.gpu_stats.wait_not_active++;

                test_data.unlock();
                sieve_mtx.unlock();

                if (set) {
                    // test_data isn't locked so do this write first before waking up batches.
                    assert( had_ready > 0 );
                    assert( test_data.active_batches == 0 );
                    test_data.active_batches = GPU_BATCHES;
                    // Mark gpu batches as active
                    for (auto& batch : gpu_batches) {
                        assert( batch.state == GPUBatch::WAITING );
                        batch.state = GPUBatch::EMPTY;
                        batch.state.notify_one();
                    }
                } else {
                    auto t0 = high_resolution_clock::now();
                    assert( had_ready == 0 );
                    sieve_data->sieves_ready.wait(0);
                    double wait = duration<double>(high_resolution_clock::now() - t0).count();
                    if (og_config.verbose >= 3) {
                        printf("Wait for sieve: o=%lu %.5f seconds\n", test_data.offset, wait);
                    }
                    test_data.lock();
                    test_data.gpu_stats.d_wait_not_active += wait;
                    test_data.unlock();
                }

                continue;
            }

            // run_gpu_thread running batches till test_data is done

            assert( state == LinearTestData::ACTIVE || state == LinearTestData::DONE );
            if (state == LinearTestData::ACTIVE ) {
                test_data.wait_for_state_and_lock(LinearTestData::DONE);
            } else {
                test_data.lock();
            }

            if (!is_running) {
                test_data.unlock();
                break;
            }

            {
                // Holding test_data.lock()
                assert( test_data.state == LinearTestData::DONE );
                assert( test_data.active_batches == 0 );
                assert( test_data.running_batches == 0 );

                // Merge all stats from gpu_stats
                for (auto& batch : gpu_batches) {
                    assert( batch.state == GPUBatch::WAITING );
                    batch.lock();
                    test_data.gpu_stats.merge(batch.stats);
                    batch.stats.reset();
                    batch.unlock();
                }

                sieve_mtx.lock();

                // TODO time this.

                // Mark m_inc tests as having been done, maybe print stats
                // TODO TBD check if these are the right stats
                test_data.stats.batches += 1;
                test_data.stats.total_m += og_config.m_inc;
                if (og_config.verbose >= 1) {
                    test_data.maybe_print_stats();
                }

                test_data.reset();

                sieve_data->go_to_next_range();
                if (stop_queue) {
                    stop_queue++;
                    if (og_config.verbose >= 2) {
                        printf("\tChanged stop_queue to %u\n", stop_queue.load());
                    }
                }

                test_data.state = LinearTestData::WAITING;


                sieve_mtx.unlock();
                test_data.unlock();
            }
        }

        // ----- cleanup
        {
            mpz_clear(K);
        }

        if (!is_running) {
            for (auto& batch : gpu_batches) {
                batch.state = GPUBatch::EMPTY;
                batch.state.notify_one();
            }
        }

        // Push note to overflow that we're done.
        for (int32_t i = 0; i < og_config.cpu_threads; i++) {
            overflow.push_to_queue(0, 0, Overflow::Type::STOP_WORKER);
        }

        if (og_config.verbose >= 1) {
            test_data.lock();
            test_data.print_stats();
            test_data.unlock();
        }

        if (og_config.verbose >= 2)
            printf("End of testing thread, Joining batch threads\n");
        // Send notifies (to wake up GPU thread and stop conditional waiting)
        for (auto& gpu_batch : gpu_batches) {
            // Should wait anyone waiting up
            gpu_batch.state = GPUBatch::EMPTY;
            gpu_batch.state.notify_all();
        }
        if (og_config.verbose >= 2)
            cout << "\tjoining gpu threads" << endl;
        size_t i = 0;
        for (auto & gpu_thread : gpu_threads) {
            gpu_thread.join();
            if (og_config.verbose >= 2)
                cout << "\tbatch gpu thread(" << i << ") joined" << endl;
            i++;
        }
        if (og_config.verbose >= 2)
            cout << "\ttesting thread done" << endl;
    } catch (const std::exception &e) {
        cout << "ERROR in testing_thread" << endl;
        cout << e.what() << endl;
        is_running = false;
    }
}


int ctrl_c_count = 0;
void signal_callback_handler(int) {
    ctrl_c_count++;
    if (stop_queue == 0) {
       cout << endl;
       cout << "Caught CTRL+C stopping, winding down work." << endl;
       cout << endl;
       stop_queue = 1;
    } else if (ctrl_c_count == 2) {
       cout << endl;
       cout << "Caught 2nd CTRL+C, is_running = false" << endl;
       is_running = false;
    } else {
       cout << endl;
       cout << "Caught 3nd CTRL+C, exit(2) now." << endl;
       exit(2);
       exit(2);
    }
}

void prime_gap_test(struct Config config) {
    // Setup test runner
    printf("\n");

    // Turn into static assert
    assert( GPU_BATCH_SIZE == 1024 || GPU_BATCH_SIZE == 2048 || GPU_BATCH_SIZE == 4096 ||
            GPU_BATCH_SIZE == 8192 || GPU_BATCH_SIZE ==16384 || GPU_BATCH_SIZE ==32768 );

    mpz_t K;
    init_K(config, K);

    gpu_state_and_checks(K, config.m_start + 1000 * config.m_inc);

    is_running = true;
    stop_queue = 0;

    // Setup CTRL+C catcher
    signal(SIGINT, signal_callback_handler);

    sieve_data = std::make_unique<SieveData>(config);

    // Setup
    {
        sieve_mtx.lock();

        auto s_start_t = high_resolution_clock::now();
        sieve_data->setup_sieve_data(false);
        if (config.verbose >= 2) {
            auto s_stop_t = high_resolution_clock::now();
            printf("\tSetup took %.1f seconds\n",
                   duration<double>(s_stop_t - s_start_t).count());
        }
        if (config.verbose >= 1)
            printf("\n");

        sieve_mtx.unlock();
    }
    std::atomic<uint8_t> setup_done{0};
    std::thread sieve_thread(run_sieve_thread, std::ref(setup_done));
    // May take a few seconds for GPUSieve to build up prime lists
    setup_done.wait(0);
    if (config.verbose >= 3)
        printf("Setup Done!\n");

    // This has output that's nicer close to the top.
    std::thread overflow_thread{run_overflow_coordinator_thread, std::ref(config)};
    usleep(10'000); // 50ms

    std::thread testing_thread{run_testing_thread, config};

    while (is_running && stop_queue <= 1) {
        usleep(50'000); // 50ms
    }

    sieve_data->sieves_ready = OPEN_SIEVES / 2;
    sieve_data->sieves_ready.notify_all();

    {
        if (config.verbose >= 2)
            cout << "Joining threads" << endl;

        sieve_thread.join();
        if (config.verbose >= 2)
            cout << "\tsieve joined" << endl;

        testing_thread.join();
        if (config.verbose >= 2)
            cout << "\ttesting joined" << endl;

        overflow_thread.join();
        if (config.verbose >= 2)
            cout << "\toverflow joined" << endl;
    }

    if (config.verbose >= 1 && is_running) {
        printf("\tresume at --mstart=%lu\n", sieve_data->config.m_start);
    }

    mpz_clear(K);
}
