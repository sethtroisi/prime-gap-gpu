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
 * Micro benchmarks of candidate kernels for gpu_sieve.cu (see sieve_notes.md), each verified against a CPU sieve.
 *
 *   make gpu_sieve_experiments && ./gpu_sieve_experiments [name filter] [max prime in millions]
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gmp.h>
#include <primesieve.hpp>

using std::vector;
using namespace std::chrono;

#define CUDA_CHECK(action) do { cudaError_t s_ = (action); if (s_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d (%s)\n", cudaGetErrorString(s_), __FILE__, __LINE__, #action); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Shared device helpers

/** first odd mi (as index mi >> 1) with (M_start + mi) * K + X = 0 mod p, via int64 % */
__device__ __forceinline__ uint32_t first_index_i64(uint64_t M_start, uint64_t X, uint32_t p, uint32_t n) {
    const int64_t prime = p;
    int64_t neg_inner = (int64_t) M_start - (int64_t) (X * n);
    int64_t temp = neg_inner % prime;
    int64_t mi_0 = temp <= 0 ? -temp : (prime - temp);
    mi_0 += (mi_0 & 1) ? 0 : prime;
    return mi_0 >> 1;
}

/** Same as above with a float reciprocal, valid for |V| < 2^45 and p > 2^23 (q error is at most 2). */
__device__ __forceinline__ uint32_t first_index_float(uint64_t M_start, uint64_t X, uint32_t p, uint32_t n) {
    const int64_t V = (int64_t) (X * n) - (int64_t) M_start;
    const float inv = 1.0f / (float) p;
    const int64_t q = (int64_t) floorf((float) V * inv);
    int64_t r = V - q * (int64_t) p;
    // r in (-3p, 3p)
    r += (r < 0) ? 2 * (int64_t) p : 0;
    r += (r < 0) ? (int64_t) p : 0;
    // r in [0, 3p)
    uint32_t mi = r;  // 3p < 2^32 for p < 1.43B
    mi = min(mi, mi - p);
    mi = min(mi, mi - p);
    mi += (mi & 1) ? 0 : p;
    return mi >> 1;
}

// ---------------------------------------------------------------------------
// Baseline kernels (copied from gpu_sieve.cu @ 6fb5ac2)

__global__ void base_small_kernel(uint64_t global_M_start, uint64_t global_M_INC_HALF, uint32_t X,
        uint8_t *composite, const uint32_t *primes, const uint32_t *neg_inv_Ks) {
    uint32_t threads = blockDim.x;
    uint32_t thread_idx = threadIdx.x;
    uint64_t thread_offset = global_M_INC_HALF * thread_idx / threads;
    uint64_t thread_offset_next = global_M_INC_HALF * (thread_idx + 1) / threads;
    uint64_t thread_M_start = global_M_start + (thread_offset << 1);
    uint32_t pi = blockIdx.x;
    const uint32_t prime = primes[pi];
    const int32_t neg_inv_K = neg_inv_Ks[pi];
    int64_t mi_0_shift = prime - (thread_M_start % prime);
    int64_t mi_0 = (X * neg_inv_K + mi_0_shift) % prime;
    mi_0 += (mi_0 & 1) ? 0 : prime;
    mi_0 >>= 1;
    for (uint32_t t = thread_offset + mi_0; t < thread_offset_next; t += prime) {
        composite[t] = true;
    }
}

/** mode: 0 = byte writes, 1 = atomicOr bits. Used for both "medium" and "large" */
template<int MODE, int LARGE, int FLOAT>
__global__ void strided_kernel(uint64_t M_start, uint32_t M_INC_HALF, uint64_t X,
        void *composite_v, uint32_t num_primes, const uint32_t *primes, const uint32_t *neg_inv_Ks) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *composite = (uint8_t*) composite_v;
    uint32_t *words = (uint32_t*) composite_v;

    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        const uint32_t p = primes[pi];
        const uint32_t n = neg_inv_Ks[pi];
        uint32_t t = FLOAT ? first_index_float(M_start, X, p, n) : first_index_i64(M_start, X, p, n);
        if (LARGE) {
            if (t < M_INC_HALF) {
                if (MODE == 0) composite[t] = true;
                else atomicOr(&words[t >> 5], 1u << (t & 31));
            }
        } else {
            for (; t < M_INC_HALF; t += p) {
                if (MODE == 0) composite[t] = true;
                else atomicOr(&words[t >> 5], 1u << (t & 31));
            }
        }
    }
}

/** Lower bound for anything that reads 8 bytes per prime */
__global__ void read_only_kernel(uint32_t num_primes, const uint32_t *primes, const uint32_t *neg_inv_Ks, uint32_t *out) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        acc += primes[pi] ^ neg_inv_Ks[pi];
    }
    out[thread_idx] = acc;
}

/** Large primes as chunk start + uint8 half gaps (5 bytes / prime), each thread handles chunks of CHUNK primes. */
template<int CHUNK>
__global__ void large_gap_kernel(uint64_t M_start, uint32_t M_INC_HALF, uint64_t X,
        uint32_t *words, uint32_t num_primes,
        const uint32_t *chunk_first, const uint8_t *half_gaps, const uint32_t *neg_inv_Ks) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t chunks = (num_primes + CHUNK - 1) / CHUNK;

    for (uint32_t ci = thread_idx; ci < chunks; ci += threads) {
        uint32_t p = chunk_first[ci];
        const uint32_t start = ci * CHUNK;
        const uint32_t end = min(start + CHUNK, num_primes);
        for (uint32_t pi = start; pi < end; pi++) {
            p += 2 * (uint32_t) half_gaps[pi];
            const uint32_t n = neg_inv_Ks[pi];
            uint32_t t = first_index_float(M_start, X, p, n);
            if (t < M_INC_HALF) {
                atomicOr(&words[t >> 5], 1u << (t & 31));
            }
        }
    }
}

/** Sieve BATCH X (X, X+2, ...) per read of (prime, inv), PACKED interleaves them in one array. */
template<int BATCH, int FLOAT, int PACKED>
__global__ void large_batch_kernel(uint64_t M_start, uint32_t M_INC_HALF, uint64_t X,
        uint32_t *words, uint32_t stride, uint32_t num_primes,
        const uint32_t *primes, const uint32_t *neg_inv_Ks, const uint64_t *packed) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        uint32_t p, n;
        if (PACKED) {
            const uint64_t pn = packed[pi];
            p = (uint32_t) pn;
            n = pn >> 32;
        } else {
            p = primes[pi];
            n = neg_inv_Ks[pi];
        }
        for (int b = 0; b < BATCH; b++) {
            const uint64_t Xb = X + 2 * b;
            uint32_t t = FLOAT ? first_index_float(M_start, Xb, p, n) : first_index_i64(M_start, Xb, p, n);
            if (t < M_INC_HALF) {
                atomicOr(&words[b * stride + (t >> 5)], 1u << (t & 31));
            }
        }
    }
}

/**
 * Copy of large_primes_batch_kernal, U primes are loaded before any math and MODE is
 * 0: atomicOr, 1: count hits, 2: racy or, 3: atomicOr if bit unset, 4: atomicOr if bit set in mask.
 */
template<int MODE, int U>
__global__ void prod_batch_kernel(uint64_t M_start, uint32_t M_INC_HALF, uint32_t batch, uint32_t X0,
        uint32_t *composite, uint32_t composite_words, uint32_t num_primes,
        const uint32_t *primes, const uint32_t *neg_inv_Ks, uint32_t *out, const uint32_t *mask = nullptr) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hits = 0;

    for (uint32_t pi = thread_idx; pi < num_primes; pi += U * threads) {
        uint32_t ps[U], ns[U];
        for (int u = 0; u < U; u++) {
            const uint32_t idx = pi + u * threads;
            const uint32_t safe = min(idx, num_primes - 1);
            ps[u] = primes[safe];
            ns[u] = neg_inv_Ks[safe];
            if (idx >= num_primes) ps[u] = 0;
        }
        for (int u = 0; u < U; u++) {
            const uint32_t prime = ps[u];
            const uint32_t neg_inv_K = ns[u];
            if (prime == 0) continue;

            const uint32_t neg_M_start = prime - (uint32_t) (M_start % prime);
            const float inv = 1.0f / (float) prime;
            const float neg_inv_K_over_p = (float) neg_inv_K * inv;

            for (uint32_t b = 0; b < batch; b++) {
                const uint32_t X = X0 + 2 * b;
                const uint32_t q = (int32_t) ((float) X * neg_inv_K_over_p - 0.05f);
                uint32_t r = X * neg_inv_K - q * prime;
                r = min(r, r - prime);
                uint32_t mi_0 = r + neg_M_start;
                mi_0 = min(mi_0, mi_0 - prime);
                mi_0 += (mi_0 & 1) ? 0 : prime;
                const uint32_t t = mi_0 >> 1;
                if (t < M_INC_HALF) {
                    if (MODE == 0) atomicOr(&composite[b * composite_words + (t >> 5)], 1u << (t & 31));
                    if (MODE == 1) hits++;
                    if (MODE == 2) composite[b * composite_words + (t >> 5)] |= 1u << (t & 31);
                    if (MODE == 3 && !(composite[b * composite_words + (t >> 5)] & (1u << (t & 31))))
                        atomicOr(&composite[b * composite_words + (t >> 5)], 1u << (t & 31));
                    if (MODE == 4 && (mask[t >> 5] & (1u << (t & 31))))
                        atomicOr(&composite[b * composite_words + (t >> 5)], 1u << (t & 31));
                }
            }
        }
    }
    if (MODE == 1) out[thread_idx] = hits;
}

/**
 * Large primes as first prime + uint8 half gaps + inverse (~5 bytes / prime), thread g owns a column
 * of ROWS consecutive primes per super block, stored row major so every load is coalesced.
 */
template<int ROWS>
__global__ void tgap_batch_kernel(uint64_t M_start, uint32_t M_INC_HALF, uint32_t batch, uint32_t X0,
        uint32_t *composite, uint32_t composite_words, uint32_t num_primes,
        const uint32_t *first, const uint8_t *gaps, const uint32_t *inv_Ks) {
    const uint32_t T = gridDim.x * blockDim.x;
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t s = 0; ; s++) {
        const uint64_t column_start = (uint64_t) s * T * ROWS + (uint64_t) g * ROWS;
        if (column_start >= num_primes) break;
        const uint32_t rows = min((uint64_t) ROWS, num_primes - column_start);

        uint32_t prime = first[s * T + g];
        for (uint32_t j = 0; j < rows; j++) {
            const uint32_t idx = (s * ROWS + j) * T + g;
            prime += 2 * (uint32_t) gaps[idx];
            const uint32_t neg_inv_K = inv_Ks[idx];

            const uint32_t neg_M_start = prime - (uint32_t) (M_start % prime);
            const float inv = 1.0f / (float) prime;
            const float neg_inv_K_over_p = (float) neg_inv_K * inv;

            for (uint32_t b = 0; b < batch; b++) {
                const uint32_t X = X0 + 2 * b;
                const uint32_t q = (int32_t) ((float) X * neg_inv_K_over_p - 0.05f);
                uint32_t r = X * neg_inv_K - q * prime;
                r = min(r, r - prime);
                uint32_t mi_0 = r + neg_M_start;
                mi_0 = min(mi_0, mi_0 - prime);
                mi_0 += (mi_0 & 1) ? 0 : prime;
                const uint32_t t = mi_0 >> 1;
                if (t < M_INC_HALF) {
                    atomicOr(&composite[b * composite_words + (t >> 5)], 1u << (t & 31));
                }
            }
        }
    }
}

/** Lower bounds for reads */
template<int MODE>
__global__ void read_only2_kernel(uint32_t num_primes, const uint32_t *primes, const uint64_t *packed, uint32_t *out) {
    const uint32_t threads = gridDim.x * blockDim.x;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (uint32_t pi = thread_idx; pi < num_primes; pi += threads) {
        if (MODE == 0) acc += primes[pi];
        else { const uint64_t pn = packed[pi]; acc += (uint32_t) pn ^ (uint32_t) (pn >> 32); }
    }
    out[thread_idx] = acc;
}

/** Segmented sieve, each block owns S bits in shared memory as bytes (BYTES=1, packed on output) or atomicOr bits. */
template<int BYTES>
__global__ void segment_kernel(uint64_t M_start, uint64_t X, uint32_t S,
        uint32_t *out_words, uint32_t num_primes, const uint32_t *primes, const uint32_t *neg_inv_Ks) {
    extern __shared__ uint32_t shared_words[];
    uint8_t *flags = (uint8_t*) shared_words;

    const uint32_t T = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t seg = blockIdx.x;
    const uint32_t shared_count = BYTES ? S / 4 : S / 32;

    for (uint32_t i = tid; i < shared_count; i += T) shared_words[i] = 0;
    __syncthreads();

    const uint64_t seg_M_start = M_start + 2ull * seg * S;
    for (uint32_t pi = tid; pi < num_primes; pi += T) {
        const uint32_t p = primes[pi];
        const uint32_t n = neg_inv_Ks[pi];
        uint32_t t = first_index_i64(seg_M_start, X, p, n);
        if (BYTES) {
            for (; t < S; t += p) flags[t] = 1;
        } else {
            for (; t < S; t += p) atomicOr(&shared_words[t >> 5], 1u << (t & 31));
        }
    }
    __syncthreads();

    uint32_t *out = out_words + seg * (S / 32);
    for (uint32_t w = tid; w < S / 32; w += T) {
        uint32_t v;
        if (BYTES) {
            // 8 bytes of 0/1 -> 8 bits
            const uint64_t *f = (const uint64_t*) (flags + 32 * w);
            const uint64_t magic = 0x0102040810204080ull;
            v  = (uint32_t) ((f[0] * magic) >> 56);
            v |= (uint32_t) ((f[1] * magic) >> 56) << 8;
            v |= (uint32_t) ((f[2] * magic) >> 56) << 16;
            v |= (uint32_t) ((f[3] * magic) >> 56) << 24;
        } else {
            v = shared_words[w];
        }
        out[w] = v;
    }
}

/** bytes -> bits (copied from gpu_sieve.cu) */
__global__ void base_compress_kernel(const uint64_t composite_size, const uint8_t *composite, uint8_t *result) {
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t size_per = 256;
    uint32_t copies = (composite_size-1) / size_per + 1;
    if (thread_idx >= copies) return;
    uint32_t start_i = thread_idx * size_per;
    uint32_t c_size = composite_size - start_i;
    uint32_t bytes = size_per < c_size ? size_per : c_size;
    uint32_t c_i = start_i;
    for (uint32_t j = 0; j < bytes; j += 8) {
        uint8_t t = (composite[c_i] | (composite[c_i+1] << 1) | (composite[c_i+2] << 2) | (composite[c_i+3] << 3)
            | (composite[c_i+4] << 4) | (composite[c_i+5] << 5) | (composite[c_i+6] << 6) | (composite[c_i+7] << 7));
        result[c_i >> 3] = t;
        c_i += 8;
    }
}

// ---------------------------------------------------------------------------
// Host

static int64_t invert_mod(int64_t a, int64_t p) {
    int64_t x = 1, x1 = 0, a1 = a, p1 = p;
    while (p1) {
        int64_t q = a1 / p1;
        int64_t t = x1; x1 = x - q * x1; x = t;
        t = p1; p1 = a1 - q * p1; a1 = t;
    }
    return x < 0 ? x + p : x;
}

struct Marks {
    uint64_t count = 0;
    uint64_t hash = 0;
    bool operator==(const Marks& o) const { return count == o.count && hash == o.hash; }
};

static Marks marks_from_bytes(const uint8_t *c, uint64_t n) {
    Marks m;
    #pragma omp parallel
    {
        uint64_t count = 0, hash = 0;
        #pragma omp for nowait
        for (uint64_t i = 0; i < n; i++) {
            if (c[i]) { count++; hash += (i + 1) * 0x9E3779B97F4A7C15ull; }
        }
        #pragma omp critical
        { m.count += count; m.hash += hash; }
    }
    return m;
}

static Marks marks_from_bits(const uint32_t *w, uint64_t n) {
    Marks m;
    for (uint64_t i = 0; i < n; i++) {
        if ((w[i >> 5] >> (i & 31)) & 1) { m.count++; m.hash += (i + 1) * 0x9E3779B97F4A7C15ull; }
    }
    return m;
}

int main(int argc, char *argv[]) {
    const char *filter = argc > 1 ? argv[1] : "";
    const uint32_t P = 353, D = 2310;
    const uint64_t M_START = 80'000'000'000ull, M_INC = 50'000'000;
    const uint64_t MAX_PRIME = (argc > 2 ? atol(argv[2]) : 1000) * 1'000'000ull;
    // These kernels predate the 2^32 support in gpu_sieve.cu (they need 2 * prime < 2^32).
    assert( MAX_PRIME <= 2'000'000'000 );
    const uint32_t BITS = M_INC / 2;
    const uint32_t REPS = 20;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const uint32_t MP_COUNT = prop.multiProcessorCount;
    const uint32_t GRID = 4 * MP_COUNT;
    const uint32_t BLOCK = 256;
    printf("GPU: %s | SMs: %u | L2 cache: %d KB | max shared/block: %zu KB (opt-in %zu KB)\n",
            prop.name, MP_COUNT, prop.l2CacheSize / 1024, prop.sharedMemPerBlock / 1024,
            prop.sharedMemPerBlockOptin / 1024);

    // ---- primes and -K^-1 mod p
    vector<uint32_t> primes, ninv;
    {
        auto t0 = high_resolution_clock::now();
        mpz_t K;
        mpz_init(K);
        mpz_primorial_ui(K, P);
        mpz_tdiv_q_ui(K, K, D);
        primesieve::generate_primes((uint64_t) P + 1, MAX_PRIME, &primes);
        ninv.resize(primes.size());
        #pragma omp parallel for schedule(static, 1 << 14)
        for (size_t i = 0; i < primes.size(); i++) {
            const uint64_t p = primes[i];
            ninv[i] = p - invert_mod(mpz_fdiv_ui(K, p), p);
        }
        mpz_clear(K);
        printf("%zu primes in (%u, %lu], setup %.1f s\n", primes.size(), P, MAX_PRIME,
                duration<double>(high_resolution_clock::now() - t0).count());
    }
    const uint32_t NUM = primes.size();
    auto index_of = [&](uint64_t v) -> uint32_t {
        return std::lower_bound(primes.begin(), primes.end(), v) - primes.begin(); };
    const uint32_t I_SMALL = 2 * GRID;
    const uint32_t I_MED = index_of(BITS);

    // gap compressed
    const uint32_t CHUNK = 32;
    vector<uint32_t> chunk_first;
    vector<uint8_t> half_gaps(NUM - I_MED + CHUNK, 0);
    for (uint32_t i = I_MED; i < NUM; i++) {
        const uint32_t j = i - I_MED;
        if (j % CHUNK == 0) chunk_first.push_back(primes[i]);
        else {
            assert( (primes[i] - primes[i-1]) / 2 < 256 );
            half_gaps[j] = (primes[i] - primes[i-1]) / 2;
        }
    }

    uint32_t *d_primes, *d_ninv, *d_words, *d_chunk_first, *d_out;
    uint8_t *d_bytes, *d_half_gaps;
    const size_t WORDS = (BITS + 31) / 32 + (1 << 16);
    const size_t BYTES = WORDS * 32;
    CUDA_CHECK(cudaMalloc(&d_primes, 4ull * NUM));
    CUDA_CHECK(cudaMalloc(&d_ninv, 4ull * NUM));
    CUDA_CHECK(cudaMalloc(&d_chunk_first, 4ull * chunk_first.size()));
    CUDA_CHECK(cudaMalloc(&d_half_gaps, half_gaps.size()));
    CUDA_CHECK(cudaMalloc(&d_words, 4 * WORDS));
    CUDA_CHECK(cudaMalloc(&d_bytes, BYTES));
    CUDA_CHECK(cudaMalloc(&d_out, 4ull * (1 << 20)));
    CUDA_CHECK(cudaMemcpy(d_primes, primes.data(), 4ull * NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ninv, ninv.data(), 4ull * NUM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_first, chunk_first.data(), 4ull * chunk_first.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_half_gaps, half_gaps.data(), half_gaps.size(), cudaMemcpyHostToDevice));

    vector<uint8_t> h_bytes(BYTES);
    vector<uint32_t> h_words(WORDS);
    vector<uint8_t> cpu(BITS);

    auto cpu_marks = [&](uint32_t i0, uint32_t i1, uint64_t X) {
        std::fill(cpu.begin(), cpu.end(), 0);
        uint8_t *c = cpu.data();
        #pragma omp parallel for schedule(static, 1 << 14)
        for (uint32_t i = i0; i < i1; i++) {
            const uint64_t p = primes[i];
            uint64_t mi = (((X % p) * ninv[i]) % p + p - (M_START % p)) % p;
            if ((mi & 1) == 0) mi += p;
            for (uint64_t t = mi >> 1; t < BITS; t += p) c[t] = 1;
        }
        return marks_from_bytes(c, BITS);
    };

    printf("\nprime index ranges: small [0, %u) p < %u | medium [%u, %u) | large [%u, %u)\n\n",
            I_SMALL, primes[I_SMALL], I_SMALL, I_MED, I_MED, NUM);
    printf("%-44s %9s %12s  %s\n", "experiment", "ms", "marks", "verify");

    // kind: 0 = bytes, 1 = bits, 2 = no output to verify
    auto run = [&](const std::string& name, int kind, uint32_t i0, uint32_t i1,
                   const std::function<void(uint64_t)>& launch) {
        if (name.find(filter) == std::string::npos) return;
        const uint64_t VX = 1234;
        auto clear = [&]() {
            if (kind == 0) CUDA_CHECK(cudaMemset(d_bytes, 0, BYTES));
            if (kind == 1) CUDA_CHECK(cudaMemset(d_words, 0, 4 * WORDS));
        };
        clear();
        launch(VX);
        CUDA_CHECK(cudaDeviceSynchronize());
        Marks got, expect;
        if (kind == 0) {
            CUDA_CHECK(cudaMemcpy(h_bytes.data(), d_bytes, BYTES, cudaMemcpyDeviceToHost));
            got = marks_from_bytes(h_bytes.data(), BITS);
        } else if (kind == 1) {
            CUDA_CHECK(cudaMemcpy(h_words.data(), d_words, 4 * WORDS, cudaMemcpyDeviceToHost));
            got = marks_from_bits(h_words.data(), BITS);
        }
        if (kind != 2) expect = cpu_marks(i0, i1, VX);

        double total = 0;
        double best = 1e9;
        for (uint32_t r = 0; r < REPS; r++) {
            clear();
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = high_resolution_clock::now();
            launch(2 + 2 * r);
            CUDA_CHECK(cudaDeviceSynchronize());
            double ms = 1000 * duration<double>(high_resolution_clock::now() - t0).count();
            total += ms;
            best = std::min(best, ms);
        }
        printf("%-44s %9.3f %12lu  %s (min %.3f)\n", name.c_str(), total / REPS, got.count,
                kind == 2 ? "-" : (got == expect ? "ok" : "MISMATCH"), best);
        fflush(stdout);
    };

    // ---- baseline
    run("base_small bytes", 0, 0, I_SMALL, [&](uint64_t X) {
        base_small_kernel<<<I_SMALL, BLOCK>>>(M_START, BITS, X, d_bytes, d_primes, d_ninv); });
    run("base_medium bytes", 0, I_SMALL, I_MED, [&](uint64_t X) {
        strided_kernel<0, 0, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_bytes, I_MED - I_SMALL, d_primes + I_SMALL, d_ninv + I_SMALL); });
    run("base_large bytes", 0, I_MED, NUM, [&](uint64_t X) {
        strided_kernel<0, 1, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_bytes, NUM - I_MED, d_primes + I_MED, d_ninv + I_MED); });
    run("base_compress", 2, 0, 0, [&](uint64_t X) {
        base_compress_kernel<<<(BITS / 256) / BLOCK + 1, BLOCK>>>(BITS, d_bytes, (uint8_t*) d_words); });

    // ---- medium
    run("medium atomic-bits", 1, I_SMALL, I_MED, [&](uint64_t X) {
        strided_kernel<1, 0, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_words, I_MED - I_SMALL, d_primes + I_SMALL, d_ninv + I_SMALL); });
    run("small+medium atomic-bits (one kernel)", 1, 0, I_MED, [&](uint64_t X) {
        strided_kernel<1, 0, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_words, I_MED, d_primes, d_ninv); });

    // ---- large
    run("large read-only (8B/prime)", 2, 0, 0, [&](uint64_t X) {
        read_only_kernel<<<GRID, BLOCK>>>(NUM - I_MED, d_primes + I_MED, d_ninv + I_MED, d_out); });
    run("large i64 atomic-bits", 1, I_MED, NUM, [&](uint64_t X) {
        strided_kernel<1, 1, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_words, NUM - I_MED, d_primes + I_MED, d_ninv + I_MED); });
    run("large float bytes", 0, I_MED, NUM, [&](uint64_t X) {
        strided_kernel<0, 1, 1><<<GRID, BLOCK>>>(M_START, BITS, X, d_bytes, NUM - I_MED, d_primes + I_MED, d_ninv + I_MED); });
    run("large float atomic-bits", 1, I_MED, NUM, [&](uint64_t X) {
        strided_kernel<1, 1, 1><<<GRID, BLOCK>>>(M_START, BITS, X, d_words, NUM - I_MED, d_primes + I_MED, d_ninv + I_MED); });
    for (uint32_t grid_mult : {1, 4, 16}) {
        char name[100];
        sprintf(name, "large float atomic-bits gaps(5B) grid=%uxMP", grid_mult * 4);
        run(name, 1, I_MED, NUM, [&](uint64_t X) {
            large_gap_kernel<CHUNK><<<grid_mult * GRID, BLOCK>>>(M_START, BITS, X, d_words, NUM - I_MED,
                    d_chunk_first, d_half_gaps, d_ninv + I_MED); });
    }
    for (uint32_t grid_mult : {4, 16}) {
        char name[100];
        sprintf(name, "large float atomic-bits grid=%uxMP", grid_mult * 4);
        run(name, 1, I_MED, NUM, [&](uint64_t X) {
            strided_kernel<1, 1, 1><<<grid_mult * GRID, BLOCK>>>(M_START, BITS, X, d_words, NUM - I_MED, d_primes + I_MED, d_ninv + I_MED); });
    }

    // ---- large: memory layout, threads, batching
    const uint32_t MAX_BATCH = 8;
    const uint32_t STRIDE = WORDS;
    uint32_t *d_batch_words;
    uint64_t *d_packed;
    CUDA_CHECK(cudaMalloc(&d_batch_words, 4ull * MAX_BATCH * STRIDE));
    CUDA_CHECK(cudaMalloc(&d_packed, 8ull * (NUM - I_MED)));
    {
        vector<uint64_t> packed(NUM - I_MED);
        for (uint32_t i = I_MED; i < NUM; i++) packed[i - I_MED] = primes[i] | ((uint64_t) ninv[i] << 32);
        CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), 8ull * packed.size(), cudaMemcpyHostToDevice));
    }
    const uint32_t NL = NUM - I_MED;
    run("large read-only 4B/prime", 2, 0, 0, [&](uint64_t X) {
        read_only2_kernel<0><<<GRID, BLOCK>>>(NL, d_primes + I_MED, d_packed, d_out); });
    run("large read-only packed 8B/prime", 2, 0, 0, [&](uint64_t X) {
        read_only2_kernel<1><<<GRID, BLOCK>>>(NL, d_primes + I_MED, d_packed, d_out); });
    for (uint32_t block : {64, 128, 256, 512, 1024}) {
        for (uint32_t blocks_per_mp : {1, 2, 4, 8, 16, 64}) {
            if (block * blocks_per_mp < 512 || block * blocks_per_mp > 16384) continue;
            char name[100];
            sprintf(name, "large shape read-only packed <<<%u*MP, %u>>>", blocks_per_mp, block);
            run(name, 2, 0, 0, [&](uint64_t X) {
                read_only2_kernel<1><<<blocks_per_mp * MP_COUNT, block>>>(NL, d_primes + I_MED, d_packed, d_out); });
        }
    }
    // Batches > 1 are timing only (d_batch_words is never cleared).
    #define BATCH_EXP(B, F, PK) { \
        char name[100]; \
        sprintf(name, "large batch=%d %s %s", B, F ? "float" : "i64", PK ? "packed" : "split"); \
        run(name, B == 1 ? 1 : 2, I_MED, NUM, [&](uint64_t X) { \
            large_batch_kernel<B, F, PK><<<GRID, BLOCK>>>(M_START, BITS, X, B == 1 ? d_words : d_batch_words, STRIDE, NL, \
                    d_primes + I_MED, d_ninv + I_MED, d_packed); }); }
    BATCH_EXP(1, 0, 0) BATCH_EXP(1, 0, 1) BATCH_EXP(1, 1, 0) BATCH_EXP(1, 1, 1)
    BATCH_EXP(2, 0, 1) BATCH_EXP(2, 1, 1)
    BATCH_EXP(4, 0, 1) BATCH_EXP(4, 1, 1)
    BATCH_EXP(8, 0, 1) BATCH_EXP(8, 1, 1)
    BATCH_EXP(4, 0, 0) BATCH_EXP(4, 1, 0)

    // What limits the production batch kernel?
    #define PROD_EXP(MODE, U, B, BPM, BLK) { \
        char name[100]; \
        sprintf(name, "prod batch=%d %s U=%d <<<%d*MP, %d>>>", B, \
                (const char*[]) {"atomic", "count ", "racy  ", "unset ", "mask  "}[MODE], U, BPM, BLK); \
        run(name, 2, I_MED, NUM, [&](uint64_t X) { \
            prod_batch_kernel<MODE, U><<<BPM * MP_COUNT, BLK>>>(M_START, BITS, B, X, d_batch_words, STRIDE, NL, \
                    d_primes + I_MED, d_ninv + I_MED, d_out); }); }
    PROD_EXP(0, 1, 1, 4, 256) PROD_EXP(0, 1, 4, 4, 256) PROD_EXP(0, 1, 8, 4, 256)
    PROD_EXP(1, 1, 1, 4, 256) PROD_EXP(1, 1, 4, 4, 256) PROD_EXP(1, 1, 8, 4, 256)
    PROD_EXP(2, 1, 8, 4, 256)
    PROD_EXP(0, 2, 8, 4, 256) PROD_EXP(0, 4, 8, 4, 256) PROD_EXP(0, 8, 8, 4, 256)
    PROD_EXP(1, 4, 8, 4, 256)
    PROD_EXP(0, 1, 8, 2, 256) PROD_EXP(0, 1, 8, 6, 256) PROD_EXP(0, 1, 8, 8, 256) PROD_EXP(0, 1, 8, 16, 256)
    PROD_EXP(0, 1, 8, 8, 128) PROD_EXP(0, 1, 8, 2, 512) PROD_EXP(0, 1, 8, 1, 1024)
    PROD_EXP(0, 4, 8, 8, 256) PROD_EXP(0, 4, 4, 4, 256) PROD_EXP(0, 4, 1, 4, 256)

    // Transposed gap layout
    for (uint32_t rows : {16, 64}) {
        const uint32_t T = GRID * BLOCK;
        const uint64_t per_sb = (uint64_t) T * rows;
        const uint32_t sbs = (NL + per_sb - 1) / per_sb;
        vector<uint32_t> h_first((size_t) sbs * T, 0), h_inv(sbs * per_sb, 0);
        vector<uint8_t> h_gaps(sbs * per_sb, 0);
        #pragma omp parallel for schedule(static, 1 << 16)
        for (uint32_t i = 0; i < NL; i++) {
            const uint32_t sb = i / per_sb, rem = i % per_sb, g = rem / rows, j = rem % rows;
            const size_t idx = ((size_t) sb * rows + j) * T + g;
            h_inv[idx] = ninv[I_MED + i];
            if (j == 0) h_first[(size_t) sb * T + g] = primes[I_MED + i];
            else h_gaps[idx] = (primes[I_MED + i] - primes[I_MED + i - 1]) / 2;
        }
        uint32_t *d_first, *d_inv;
        uint8_t *d_gaps;
        CUDA_CHECK(cudaMalloc(&d_first, 4 * h_first.size()));
        CUDA_CHECK(cudaMalloc(&d_inv, 4 * h_inv.size()));
        CUDA_CHECK(cudaMalloc(&d_gaps, h_gaps.size()));
        CUDA_CHECK(cudaMemcpy(d_first, h_first.data(), 4 * h_first.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv, h_inv.data(), 4 * h_inv.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gaps, h_gaps.data(), h_gaps.size(), cudaMemcpyHostToDevice));
        for (uint32_t batch : {1, 4, 8}) {
            char name[100];
            sprintf(name, "tgap rows=%u batch=%u", rows, batch);
            #define TGAP(ROWS) tgap_batch_kernel<ROWS><<<GRID, BLOCK>>>(M_START, BITS, batch, X, \
                    batch == 1 ? d_words : d_batch_words, STRIDE, NL, d_first, d_gaps, d_inv)
            run(name, batch == 1 ? 1 : 2, I_MED, NUM, [&](uint64_t X) {
                if (rows == 16) TGAP(16); else TGAP(64); });
        }
        CUDA_CHECK(cudaFree(d_first)); CUDA_CHECK(cudaFree(d_inv)); CUDA_CHECK(cudaFree(d_gaps));
    }

    // Skip atomics for bits that don't matter
    {
        vector<uint32_t> fill(MAX_BATCH * STRIDE);
        uint64_t rng = 88172645463325252ull;
        auto random_words = [&](vector<uint32_t> &v, double density) {
            for (auto &w : v) {
                w = 0;
                for (int i = 0; i < 32; i++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    w |= (uint32_t) ((rng >> 11) * 0x1p-53 < density) << i;
                }
            }
        };
        random_words(fill, 0.86);
        CUDA_CHECK(cudaMemcpy(d_batch_words, fill.data(), 4ull * fill.size(), cudaMemcpyHostToDevice));
        PROD_EXP(0, 1, 8, 4, 256)
        PROD_EXP(3, 1, 8, 4, 256) PROD_EXP(3, 1, 4, 4, 256)

        uint32_t *d_mask;
        CUDA_CHECK(cudaMalloc(&d_mask, 4ull * STRIDE));
        vector<uint32_t> mask(STRIDE);
        for (double density : {0.42, 0.25, 0.10}) {
            random_words(mask, density);
            CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), 4ull * mask.size(), cudaMemcpyHostToDevice));
            for (int batch : {4, 8}) {
                char name[100];
                sprintf(name, "prod batch=%d active-mask %.2f", batch, density);
                run(name, 2, I_MED, NUM, [&](uint64_t X) {
                    prod_batch_kernel<4, 1><<<4 * MP_COUNT, 256>>>(M_START, BITS, batch, X, d_batch_words, STRIDE, NL,
                            d_primes + I_MED, d_ninv + I_MED, d_out, d_mask); });
            }
        }
    }

    // D2H copy of the bit packed composite to pinned / pageable memory
    {
        void *pinned;
        CUDA_CHECK(cudaMallocHost(&pinned, BITS / 8));
        run("copy D2H pinned", 2, 0, 0, [&](uint64_t X) {
            CUDA_CHECK(cudaMemcpy(pinned, d_words, BITS / 8, cudaMemcpyDeviceToHost)); });
        run("copy D2H pageable", 2, 0, 0, [&](uint64_t X) {
            CUDA_CHECK(cudaMemcpy(h_words.data(), d_words, BITS / 8, cudaMemcpyDeviceToHost)); });
    }

    // ---- segmented
    CUDA_CHECK(cudaFuncSetAttribute(segment_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(segment_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    for (int bytes : {1, 0}) {
        for (uint32_t S : {1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19}) {
            const uint32_t shared = bytes ? S : S / 8;
            if (shared > 64 * 1024) continue;
            for (uint32_t T : {256, 1024}) {
                // primes < S and primes < 4S
                for (uint32_t mult : {1, 4}) {
                    const uint32_t count = index_of((uint64_t) mult * S);
                    const uint32_t segments = (BITS + S - 1) / S;
                    char name[100];
                    sprintf(name, "segment %s S=%uK T=%u p<%uK", bytes ? "bytes" : "atomic", S >> 10, T, (mult * S) >> 10);
                    run(name, 1, 0, count, [&](uint64_t X) {
                        if (bytes) segment_kernel<1><<<segments, T, shared>>>(M_START, X, S, d_words, count, d_primes, d_ninv);
                        else       segment_kernel<0><<<segments, T, shared>>>(M_START, X, S, d_words, count, d_primes, d_ninv);
                    });
                }
            }
        }
    }

    // ---- rest of medium (primes above the segmented limit)
    for (uint32_t limit : {1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22}) {
        const uint32_t i0 = index_of(limit);
        char name[100];
        sprintf(name, "medium atomic-bits p>=%uK", limit >> 10);
        run(name, 1, i0, I_MED, [&](uint64_t X) {
            strided_kernel<1, 0, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_words, I_MED - i0, d_primes + i0, d_ninv + i0); });
        sprintf(name, "medium bytes p>=%uK", limit >> 10);
        run(name, 0, i0, I_MED, [&](uint64_t X) {
            strided_kernel<0, 0, 0><<<GRID, BLOCK>>>(M_START, BITS, X, d_bytes, I_MED - i0, d_primes + i0, d_ninv + i0); });
    }
    return 0;
}
