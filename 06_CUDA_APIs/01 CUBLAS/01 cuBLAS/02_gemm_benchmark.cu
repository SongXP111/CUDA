#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Matrix dimensions (2048 x 2048)
const int M = 2048;
const int K = 2048;
const int N = 2048;

// Naive CPU matrix multiplication
void cpu_matmul(const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    printf("Matrix dimensions: M = %d, K = %d, N = %d\n\n", M, K, N);
    
    // Allocate CPU memory
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_cpu = (float*)malloc(bytes_C);
    float *h_C_gpu_s = (float*)malloc(bytes_C);
    float *h_C_gpu_h = (float*)malloc(bytes_C);
    
    // Initialize data
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    // ----------------------------------------------------
    // 1. CPU Benchmarking
    // ----------------------------------------------------
    printf("Running CPU Benchmarking... (This might take a few seconds)\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_matmul(h_A, h_B, h_C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("CPU time: %.2f ms\n\n", cpu_time);
    
    // ----------------------------------------------------
    // GPU Environment Setup
    // ----------------------------------------------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Query VRAM before allocations
    size_t free_before, total_before;
    CHECK_CUDA(cudaMemGetInfo(&free_before, &total_before));
    
    // Allocate GPU FP32 memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    
    // Query VRAM after FP32 allocations
    size_t free_after_fp32, total_after_fp32;
    CHECK_CUDA(cudaMemGetInfo(&free_after_fp32, &total_after_fp32));
    double fp32_vram_used = (double)(free_before - free_after_fp32) / (1024.0 * 1024.0);
    printf("GPU Memory after FP32 allocation: Free = %.2f MB, Used = %.2f MB\n", 
           (double)free_after_fp32 / (1024.0 * 1024.0), fp32_vram_used);
    
    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // ----------------------------------------------------
    // 2. cuBLAS SGEMM (FP32) Benchmark
    // ----------------------------------------------------
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    printf("Warming up cuBLAS SGEMM...\n");
    for (int i = 0; i < 5; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Profile
    printf("Benchmarking cuBLAS SGEMM (FP32)...\n");
    CHECK_CUDA(cudaEventRecord(start, 0));
    const int runs = 20;
    for (int i = 0; i < runs; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float sgemm_total_time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&sgemm_total_time, start, stop));
    float sgemm_avg_time = sgemm_total_time / runs;
    printf("cuBLAS SGEMM Average time: %.4f ms\n\n", sgemm_avg_time);
    
    // Copy FP32 result back for validation
    CHECK_CUDA(cudaMemcpy(h_C_gpu_s, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    // ----------------------------------------------------
    // 3. cuBLAS HGEMM (FP16) Benchmark
    // ----------------------------------------------------
    // Convert to half precision on CPU
    half *h_A_h = (half*)malloc(M * K * sizeof(half));
    half *h_B_h = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A_h[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_h[i] = __float2half(h_B[i]);
    
    // Allocate FP16 memory on GPU
    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));
    
    // Query VRAM after FP16 allocations
    size_t free_after_fp16, total_after_fp16;
    CHECK_CUDA(cudaMemGetInfo(&free_after_fp16, &total_after_fp16));
    double fp16_vram_used = (double)(free_after_fp32 - free_after_fp16) / (1024.0 * 1024.0);
    printf("GPU Memory after FP16 allocation: Free = %.2f MB, Used = %.2f MB\n", 
           (double)free_after_fp16 / (1024.0 * 1024.0), fp16_vram_used);
    
    // Copy FP16 data to GPU
    CHECK_CUDA(cudaMemcpy(d_A_h, h_A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, h_B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    
    // Warmup
    printf("Warming up cuBLAS HGEMM...\n");
    for (int i = 0; i < 5; i++) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Profile
    printf("Benchmarking cuBLAS HGEMM (FP16)...\n");
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < runs; i++) {
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float hgemm_total_time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&hgemm_total_time, start, stop));
    float hgemm_avg_time = hgemm_total_time / runs;
    printf("cuBLAS HGEMM Average time: %.4f ms\n\n", hgemm_avg_time);
    
    // Copy FP16 result back and convert to float for validation
    half *h_C_gpu_h_half = (half*)malloc(M * N * sizeof(half));
    CHECK_CUDA(cudaMemcpy(h_C_gpu_h_half, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) h_C_gpu_h[i] = __half2float(h_C_gpu_h_half[i]);
    
    // ----------------------------------------------------
    // Verification
    // ----------------------------------------------------
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(h_C_cpu[i] - h_C_gpu_s[i]) > 1e-2 || std::abs(h_C_cpu[i] - h_C_gpu_h[i]) > 1e-1) {
            correct = false;
            printf("Mismatch at %d: CPU=%.4f, SGEMM=%.4f, HGEMM=%.4f\n", i, h_C_cpu[i], h_C_gpu_s[i], h_C_gpu_h[i]);
            break;
        }
    }
    if (correct) {
        printf("Verification SUCCESS: All results are correct!\n\n");
    } else {
        printf("Verification FAILED!\n\n");
    }
    
    // ----------------------------------------------------
    // Benchmark Summary
    // ----------------------------------------------------
    printf("=================== Benchmark Summary ===================\n");
    printf("Matrix dimensions   : %d x %d x %d\n", M, K, N);
    printf("CPU (Naive) time    : %.2f ms\n", cpu_time);
    printf("cuBLAS SGEMM (FP32) : %.4f ms (Speedup vs CPU: %.1fx)\n", sgemm_avg_time, cpu_time / sgemm_avg_time);
    printf("cuBLAS HGEMM (FP16) : %.4f ms (Speedup vs CPU: %.1fx, vs SGEMM: %.1fx)\n", 
           hgemm_avg_time, cpu_time / hgemm_avg_time, sgemm_avg_time / hgemm_avg_time);
    printf("=========================================================\n");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu_s); free(h_C_gpu_h);
    free(h_A_h); free(h_B_h); free(h_C_gpu_h_half);
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h)); CHECK_CUDA(cudaFree(d_B_h)); CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return 0;
}
