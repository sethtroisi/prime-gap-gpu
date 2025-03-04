// Copyright 2025 Seth Troisi
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

#include <cstdint>
#include <bitset>
#include <vector>

using std::vector;

class Cached {
    public:
        // mi such that gcd(m_start + mi, D) = 1
        // TODO: Can I change this to m_inc as a uint8_t?
        vector<uint32_t> valid_mi;
        // valid_mi.size()
        size_t valid_ms;

        /**
         * m_reindex[mi] = mii (index into composite) if coprime
         * -1 if gcd(ms + i, D) > 1
         *
         * This is potentially very large use is_m_coprime and is_m_coprime2310
         * to pre check it's a coprime mi before doing the L3/RAM lookup.
         */
        vector<int32_t> m_reindex;
        // if gcd(ms + mi, D) = 1
        // TODO try with dynamic_bitset
        vector<bool> is_m_coprime;
        /**
         * is_m_coprime2310[i] = (i, D') == 1
         * D' = gcd(2310, D)
         * first 2310 values.
         * vector<bool> seems faster than char [2310]
         */
        std::bitset<2310> is_m_coprime2310;


        // X which are coprime to K
        vector<uint32_t> coprime_X;
        // reindex composite[m][X] for composite[m_reindex[m]][x_reindex[X]]
        // Special 0'th entry stands for all not coprime
        vector<uint32_t> x_reindex;

        uint64_t composite_line_size;

        int32_t K_mod2310;

        // is_comprime2310[i] = (i % 2) && (i % 3) && (i % 5) && (i % 7) && (i % 11)
        vector<char> is_coprime2310;

        Cached(const struct Config& config, const mpz_t &K);
};
