#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d in block %d!\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    hello<<<2, 4>>>();  // 2 个 block，每个 block 4 个线程
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}
