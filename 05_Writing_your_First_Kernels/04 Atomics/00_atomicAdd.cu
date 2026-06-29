#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, err, cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// Kernel without atomics (incorrect)
__global__ void incrementCounterNonAtomic(int* counter) {
    // not locked
    int old = *counter;
    int new_value = old + 1;
    // not unlocked
    *counter = new_value;
}

// Kernel with atomics (correct)
__global__ void incrementCounterAtomic(int* counter) {
    int a = atomicAdd(counter, 1);
}

int main() {
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_counterNonAtomic, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_counterAtomic, sizeof(int)));

    // Copy initial counter values to device
    CUDA_CHECK(cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernels
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    // Free device memory
    CUDA_CHECK(cudaFree(d_counterNonAtomic));
    CUDA_CHECK(cudaFree(d_counterAtomic));

    return 0;
}