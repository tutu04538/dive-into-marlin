// SPDX-License-Identifier: Apache-2.0
// Cache-hint PTX sequence derived from IST-DASLab/marlin,
// Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s (%d)\n", #call, cudaGetErrorString(e), int(e)); \
    return 1; } } while (0)

template<bool Hint>
__global__ void copy_probe(const int4* input, int4* output) {
    __shared__ int4 buffer[256];
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(&buffer[threadIdx.x]));
    if constexpr (Hint) {
        asm volatile(
            "{\n"
            " .reg .b64 p;\n"
            " createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
            " cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, p;\n"
            "}\n" :: "r"(smem), "l"(&input[threadIdx.x]));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(smem), "l"(&input[threadIdx.x]));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
    __syncthreads();
    output[threadIdx.x] = buffer[threadIdx.x];
}

int main(int argc, char** argv) {
    const bool hint = argc > 1 && std::atoi(argv[1]) != 0;
    int4 *input, *output;
    std::vector<int4> host(256), result(256);
    for (int i = 0; i < 256; ++i) host[i] = make_int4(i, i+1, i+2, i+3);
    CHECK(cudaMalloc(&input, host.size()*sizeof(int4)));
    CHECK(cudaMalloc(&output, host.size()*sizeof(int4)));
    CHECK(cudaMemcpy(input, host.data(), host.size()*sizeof(int4), cudaMemcpyHostToDevice));
    if (hint) copy_probe<true><<<1,256>>>(input, output);
    else copy_probe<false><<<1,256>>>(input, output);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result.data(), output, result.size()*sizeof(int4), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 256; ++i) {
        if (result[i].x!=i || result[i].y!=i+1 || result[i].z!=i+2 || result[i].w!=i+3) {
            std::fprintf(stderr, "Mismatch at %d\n", i);
            return 2;
        }
    }
    CHECK(cudaFree(input));
    CHECK(cudaFree(output));
    std::printf("hint=%d: copy PASS\n", hint);
}
