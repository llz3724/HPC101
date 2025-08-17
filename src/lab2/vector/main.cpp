#include "buffer.h"
#include "tile.h"
#include "reshape.h"
#include <unistd.h>

#include <iostream>
#include <cstring>
#include <chrono>

#define M 12
#define K 32
#define N 32

void naive_gemm(uint8_t* A, int8_t* B, uint32_t* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int kk = 0; kk < k * 4; kk++) {
                C[i * n + j] += A[i * k * 4 + kk] * B[j * k * 4 + kk];
            }
        }
    }
}

int main() {
    uint8_t* A = new uint8_t[M * K * 4];
    int8_t* B = new int8_t[N * K * 4];
    int8_t* B_reshape = new int8_t[N * K * 4];

    uint32_t* C = new uint32_t[M * N];
    uint32_t* C_ref = new uint32_t[M * N];

    init_buffer(A, M, K * 4);
    init_buffer(B, N, K * 4);

    // Naive Implementation (Baseline)
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 100000; iter ++) {
        memset(C_ref, 0, sizeof(uint32_t) * M * N);
        naive_gemm(A, B, C_ref, M, N, K);
    }

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end_time - start_time;
    std::cout << "Time: " << duration1.count() << " s" << std::endl;

    // Optimized
    // Write your code here
    // You can only modify the code between 'start' and 'end'

    start_time = std::chrono::high_resolution_clock::now();
    set_tiledata_use();

    // Initialize AMX tile
    // start

    // end

    for (int iter = 0; iter < 100000; iter ++) {
        memset(C, 0, sizeof(uint32_t) * M * N);
        // reshape B
        reshape((uint8_t*)B, N, K * 4, (uint8_t*)B_reshape);

        // !!!Choose a matrix multiplication from the following to achieve!!!
        // do AVX GEMM / AMX GEMM
        // start

        // end
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end_time - start_time;
    std::cout << "Time: " << duration2.count() << " s" << std::endl;

    // Verify the answer
    check_result(C, C_ref, M, N);
    double speedup = duration1.count() / duration2.count();
    std::cout << "Speedup: " << speedup << std::endl;

    return 0;
}
