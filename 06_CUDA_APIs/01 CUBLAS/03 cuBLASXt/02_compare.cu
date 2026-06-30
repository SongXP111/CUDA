#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }
#define CHECK_CUBLAS(call) { cublasStatus_t status = call; if (status != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS error: %d, line %d\n", status, __LINE__); exit(1); } }

void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool compareResults(float* result1, float* result2, int size, float tolerance) {
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(result1[i] - result2[i]);
        float max_val = std::max(std::abs(result1[i]), std::abs(result2[i]));
        if (max_val > 0.0f && (diff / max_val > tolerance)) {
            std::cout << "Results do not match at index " << i << std::endl;
            std::cout << "CUBLAS: " << result1[i] << ", CUBLAS-XT: " << result2[i] << std::endl;
            std::cout << "Relative difference: " << diff / max_val << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int M = 16384;
    int N = 16384;
    int K = 16384;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cublas = (float*)malloc(size_C);
    float *h_C_cublasxt = (float*)malloc(size_C);

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    const int num_runs = 5;
    std::vector<double> cublas_times;
    std::vector<double> cublasxt_times;

    double h2d_time = 0.0;
    double d2h_time = 0.0;

    // CUBLAS
    {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMalloc(&d_B, size_B));
        CHECK_CUDA(cudaMalloc(&d_C, size_C));

        // Measure Host-to-Device Copy Time
        auto h2d_start = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
        auto h2d_end = std::chrono::high_resolution_clock::now();
        h2d_time = std::chrono::duration<double>(h2d_end - h2d_start).count();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Warmup run
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark runs
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            cublas_times.push_back(diff.count());
            std::cout << "CUBLAS run " << i+1 << " time: " << diff.count() << " seconds" << std::endl;
        }

        // Measure Device-to-Host Copy Time
        auto d2h_start = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, size_C, cudaMemcpyDeviceToHost));
        auto d2h_end = std::chrono::high_resolution_clock::now();
        d2h_time = std::chrono::duration<double>(d2h_end - d2h_start).count();

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUBLAS(cublasDestroy(handle));
    }

    // CUBLAS-XT
    {
        cublasXtHandle_t handle;
        CHECK_CUBLAS(cublasXtCreate(&handle));

        int devices[1] = {0};
        CHECK_CUBLAS(cublasXtDeviceSelect(handle, 1, devices));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Warmup run
        CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, h_B, N, h_A, K, &beta, h_C_cublasxt, N));

        // Benchmark runs
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, h_B, N, h_A, K, &beta, h_C_cublasxt, N));
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            cublasxt_times.push_back(diff.count());
            std::cout << "CUBLAS-XT run " << i+1 << " time: " << diff.count() << " seconds" << std::endl;
        }

        CHECK_CUBLAS(cublasXtDestroy(handle));
    }

    // Calculate average times
    double avg_cublas = 0.0, avg_cublasxt = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        avg_cublas += cublas_times[i];
        avg_cublasxt += cublasxt_times[i];
    }
    avg_cublas /= num_runs;
    avg_cublasxt /= num_runs;

    // Calculate Metrics
    double total_flops = 2.0 * M * N * K; // FMA count
    double cublas_tflops = (total_flops * 1e-12) / avg_cublas;
    double cublasxt_tflops = (total_flops * 1e-12) / avg_cublasxt;

    double h2d_size_gb = (double)(size_A + size_B) / (1024.0 * 1024.0 * 1024.0);
    double d2h_size_gb = (double)(size_C) / (1024.0 * 1024.0 * 1024.0);
    double h2d_bandwidth = h2d_size_gb / h2d_time;
    double d2h_bandwidth = d2h_size_gb / d2h_time;

    std::cout << "\n================= Benchmark Metrics =================" << std::endl;
    std::cout << "Matrix Dimensions       : " << M << " x " << K << " x " << N << std::endl;
    std::cout << "Total Operations        : " << total_flops * 1e-12 << " TFLOPs" << std::endl;
    std::cout << "Average CUBLAS time     : " << avg_cublas << " seconds" << std::endl;
    std::cout << "Achieved CUBLAS TFLOPS  : " << cublas_tflops << " TFLOPS" << std::endl;
    std::cout << "Average CUBLAS-XT time  : " << avg_cublasxt << " seconds" << std::endl;
    std::cout << "Achieved CUBLAS-XT TFLOPS: " << cublasxt_tflops << " TFLOPS" << std::endl;
    std::cout << "---------------- PCIe Bandwidth (FP32) --------------" << std::endl;
    std::cout << "H2D Copy size           : " << h2d_size_gb << " GB" << std::endl;
    std::cout << "H2D Copy time           : " << h2d_time << " seconds" << std::endl;
    std::cout << "H2D Copy bandwidth      : " << h2d_bandwidth << " GB/s" << std::endl;
    std::cout << "D2H Copy size           : " << d2h_size_gb << " GB" << std::endl;
    std::cout << "D2H Copy time           : " << d2h_time << " seconds" << std::endl;
    std::cout << "D2H Copy bandwidth      : " << d2h_bandwidth << " GB/s" << std::endl;
    std::cout << "=====================================================\n" << std::endl;

    // Verify results
    float tolerance = 1e-4f;
    bool results_match = compareResults(h_C_cublas, h_C_cublasxt, M * N, tolerance);
    if (results_match) {
        std::cout << "Results match within tolerance." << std::endl;
    } else {
        std::cout << "Results do not match within tolerance." << std::endl;
    }

    free(h_A);
    free(h_B);
    free(h_C_cublas);
    free(h_C_cublasxt);

    return 0;
}
