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

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>

#include <gmp.h>

#include "gap_common.h"
#include "gap_stats.h"


struct Overflow {
    uint64_t m;

    /**
     * NEXT_PRIME: m * K + x, x >= d
     * SPOT_CHECK: m * X + d
     * PREV_PRIME: d is next_gap (e.g. positive x such that m * K + x is prime)
     */
    uint32_t d;

    enum class Type : uint8_t {
        NEXT_PRIME, PREV_PRIME, SPOT_CHECK, STOP_WORKER
    } type;

    Overflow(uint64_t m, uint32_t d, Type type) : m(m), d(d), type(type) {}
};

class OverflowQueue {
    public:
        void lock() {
            while (flag.exchange(1) == 1) {
                flag.wait(1, std::memory_order_relaxed);
            }
        }

        void unlock() {
            assert(flag.load() == 1); // locked (because we own it)
            flag = 0;
            flag.notify_one();
        }

        // TODO would be nice for this to be read only
        std::atomic<uint32_t> size;
        std::deque<Overflow> queue;

        /** should call notify_one most of the time this is called */
        void push_to_queue(uint64_t m, uint32_t d, Overflow::Type type) {
            lock();
            queue.emplace_back(m, d, type);
            size++;
            unlock();
            size.notify_one();
        }

        Overflow wait_and_get() {
            // TODO how to handle stop_queue and is_running
            while (true) {
                size.wait(0);
                lock();
                if (size > 0) {
                    assert( queue.size() == size.load() );
                    Overflow e = queue.front();
                    queue.pop_front();
                    size--;
                    unlock();
                    return e;
                }
                unlock();
            }
        }

    private:
        // For signaling, must be owned to change state.
        /**
         * :wait(0) -> unlock
         * -> set to 1 to lock with a check?
         */
        std::atomic<int> flag;
};

/**
 * TODO just pass this to run_cpu_overflow_thread
 * Also needed in gpu_testing.cu
 */
extern OverflowQueue overflow;

void run_overflow_coordinator_thread(const struct Config og_config);
