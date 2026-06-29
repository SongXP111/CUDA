#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, err, cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMul(float* A, float* B, float* C, int N) {
    nvtxRangePush("Matrix Multiplication");
    
    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    nvtxRangePush("Memory Allocation");
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    nvtxRangePop();

    nvtxRangePop();  // End of Matrix Multiplication
}

int main() {
    const int N = 1024;
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    // Initialize matrices A and B here...

    matrixMul(A, B, C, N);

    // Use result in C...

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}