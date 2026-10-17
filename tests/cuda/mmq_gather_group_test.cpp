// mmq_gather_group_test - the one-launch group gather (gather_native_group) against one gather_native per expert,
// byte for byte, on random native expert blobs at scattered 256-byte aligned addresses (as cache slots are).
// Cases: a full group of 16, a partial last group (5), a single expert, and a flush that starts inside the group
// (first > 0).  The whole group buffers, including slots the call must not touch, are compared.  Optionally times
// both on Gyro-M-sized blobs (TQ2_T: 1,305,600 B, gate/up halves 435,200 B, down 435,200 B):
//     mmq_gather_group_test [--time=ITERS]
// Exit 0: identical; 1: a mismatch; 77: no CUDA device.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;

int main(int argc, char** argv) {
    int iters = 0;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--time=", 7) == 0) iters = std::atoi(argv[i] + 7);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no CUDA device\n"); return 77; }
    const size_t half = 435200, down = 435200, up_off = half, down_off = 2 * half, blob = 2 * half + down;
    const size_t gu_stride = 2 * half, d_stride = down, tail = 4096;
    const int nblobs = 40;
    // an arena of scattered slots: blob i at a random 256-aligned offset, never overlapping
    std::vector<size_t> off(nblobs);
    size_t at = 0;
    std::mt19937_64 rng(12345);
    for (int i = 0; i < nblobs; ++i) {
        at += (rng() % 8) * 256;
        off[i] = at;
        at += (blob + 255) / 256 * 256;
    }
    uint8_t* arena = nullptr;
    if (cudaMalloc(&arena, at) != cudaSuccess) { std::printf("alloc failed\n"); return 1; }
    {
        std::vector<uint8_t> h(at);
        for (auto& b : h) b = (uint8_t) rng();
        cudaMemcpy(arena, h.data(), at, cudaMemcpyHostToDevice);
    }
    const size_t gbytes = (size_t) mmq::kGatherGroupMax * gu_stride + tail, dbytes = (size_t) mmq::kGatherGroupMax * d_stride + tail;
    uint8_t *gu_a, *d_a, *gu_b, *d_b;
    cudaMalloc(&gu_a, gbytes); cudaMalloc(&d_a, dbytes); cudaMalloc(&gu_b, gbytes); cudaMalloc(&d_b, dbytes);
    struct Case { const char* name; int first, n; };
    const Case cases[] = {{"full group (16)", 0, 16}, {"partial last group (5)", 0, 5}, {"one expert", 0, 1},
                          {"flush inside a group [6, 13)", 6, 13}};
    int fails = 0;
    for (const Case& c : cases) {
        mmq::GatherGroup g;
        for (int q = 0; q < mmq::kGatherGroupMax; ++q) g.blob[q] = arena + off[(q * 7 + 3) % nblobs];
        g.first = c.first;
        g.n = c.n;
        // both buffers start from the same pattern, so bytes outside [first, n) must stay equal too
        cudaMemset(gu_a, 0xA5, gbytes); cudaMemset(d_a, 0x5A, dbytes);
        cudaMemset(gu_b, 0xA5, gbytes); cudaMemset(d_b, 0x5A, dbytes);
        for (int q = c.first; q < c.n; ++q)
            mmq::gather_native(g.blob[q], g.blob[q] + up_off, half, g.blob[q] + down_off, down, gu_a + q * gu_stride,
                               d_a + q * d_stride, nullptr);
        const bool ok_call = mmq::gather_native_group(g, up_off, half, down_off, down, gu_b, gu_stride, d_b, d_stride, nullptr);
        cudaDeviceSynchronize();
        std::vector<uint8_t> ha(gbytes), hb(gbytes), da(dbytes), db(dbytes);
        cudaMemcpy(ha.data(), gu_a, gbytes, cudaMemcpyDeviceToHost); cudaMemcpy(hb.data(), gu_b, gbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(da.data(), d_a, dbytes, cudaMemcpyDeviceToHost); cudaMemcpy(db.data(), d_b, dbytes, cudaMemcpyDeviceToHost);
        const bool same = ok_call && ha == hb && da == db;
        if (!same) ++fails;
        std::printf("%-30s %s%s\n", c.name, same ? "identical" : "MISMATCH", ok_call ? "" : " (group call refused)");
    }
    if (iters > 0) {
        mmq::GatherGroup g;
        for (int q = 0; q < mmq::kGatherGroupMax; ++q) g.blob[q] = arena + off[(q * 7 + 3) % nblobs];
        g.first = 0; g.n = mmq::kGatherGroupMax;
        auto time = [&](bool grouped) {
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) {
                if (grouped) mmq::gather_native_group(g, up_off, half, down_off, down, gu_b, gu_stride, d_b, d_stride, nullptr);
                else
                    for (int q = 0; q < g.n; ++q)
                        mmq::gather_native(g.blob[q], g.blob[q] + up_off, half, g.blob[q] + down_off, down,
                                           gu_a + q * gu_stride, d_a + q * d_stride, nullptr);
            }
            cudaDeviceSynchronize();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        };
        time(false); time(true);   // warm
        const double per = time(false), grp = time(true), gb = 2.0 * 16 * blob / 1e9;
        std::printf("16 experts: per-expert gathers %.1f us (%.0f GB/s moved), one group launch %.1f us (%.0f GB/s)\n",
                    per, gb / (per * 1e-6), grp, gb / (grp * 1e-6));
    }
    std::printf("mmq_gather_group_test: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
