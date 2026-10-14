#include "marlin_cuda_kernel.cu"
#include <vector>
#include <cstdio>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s (%d)\n", #call, cudaGetErrorString(e), int(e)); \
    return 1; } } while (0)

int main() {
    constexpr int m=16, n=256, k=512, group=128;
    std::vector<half> a(m*k, __float2half(1.f)), s(k/group*n, __float2half(1.f)), c(m*n);
    std::vector<int> b(k*n/8, int(0x99999999u));
    half *da, *ds, *dc;
    int *db, *workspace;
    CHECK(cudaMalloc(&da, a.size()*sizeof(half)));
    CHECK(cudaMalloc(&ds, s.size()*sizeof(half)));
    CHECK(cudaMalloc(&dc, c.size()*sizeof(half)));
    CHECK(cudaMalloc(&db, b.size()*sizeof(int)));
    CHECK(cudaMalloc(&workspace, n/128*16*sizeof(int)));
    CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(half),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ds,s.data(),s.size()*sizeof(half),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dc,0,c.size()*sizeof(half)));
    CHECK(cudaMemset(workspace,0,n/128*16*sizeof(int)));
    int status=marlin_cuda(da,db,dc,ds,m,n,k,workspace,group,0,0,128,128,-1,16);
    std::printf("marlin_cuda return: %d\n",status);
    if(status) return 2;
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(c.data(),dc,c.size()*sizeof(half),cudaMemcpyDeviceToHost));
    for(int i=0;i<m*n;++i) if(__half2float(c[i])!=float(k)) {
        std::fprintf(stderr,"Incorrect output at %d: %g expected %d\n",i,__half2float(c[i]),k);
        return 3;
    }
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc)); CHECK(cudaFree(ds)); CHECK(cudaFree(workspace));
    std::puts("standalone Marlin: PASS (all outputs equal 512)");
}
