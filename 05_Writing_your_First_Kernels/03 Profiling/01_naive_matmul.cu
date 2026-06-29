#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, err, cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
int main() {

    // Define matrix dimensions
    const int M = 1024; // Number of rows in A and C
    const int N = 1024; // Number of columns in B and C
    const int K = 1024; // Number of columns in A and rows in B

    // Calculate matrix sizes in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));


    // Kernel launch code
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize device
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));



    return 0;

}