// #include <cstdlib>
// #include <iostream>

// #include "judger.h"

// extern "C" int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol);

// int main(int argc, char* argv[]) {
//     int world_size, world_rank = 0;
//     // When using MPI, please remember to initialize here

//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <input_data>" << std::endl;
//         return -1;
//     }
//     // Read data from file
//     std::string filename = argv[1];

//     // N: size of matrix A (N x N)
//     // A: matrix A
//     // b: vector b
//     // x: initial guess of solution
//     int N;
//     double *A = nullptr, *b = nullptr, *x = nullptr;

//     // Read data from file
//     if (world_rank == 0) {
//         read_data(filename, &N, &A, &b, &x);
//     }

//     // Call BiCGSTAB function
//     auto start = std::chrono::high_resolution_clock::now();
//     int iter   = bicgstab(N, A, b, x, MAX_ITER, TOL);
//     auto end   = std::chrono::high_resolution_clock::now();

//     // Check the result
//     if (world_rank == 0) {
//         auto duration = end - start;
//         judge(iter, duration, N, A, b, x);
//     }

//     // Free allocated memory
//     free(A);
//     free(b);
//     free(x);

//     // When using MPI, please remember to finalize here
//     return 0;
// }
#include <cstdlib>
#include <iostream>
#include <vector>   // For convenient buffer management
#include <chrono>   // Unchanged from original
#include <mpi.h>    // Added for MPI functionality

#include "judger.h"

// The function signature is updated to the new MPI-aware solver
extern "C" int bicgstab_mpi(int N, int local_N, double* local_A, double* local_b, double* local_x,
                            int max_iter, double tol, int rank, int* counts, int* displs);

int main(int argc, char* argv[]) {
    // Original variables
    int world_size, world_rank;

    // --- MPI logic injected as requested by comments ---
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // 在这里加入探针，每个进程都会打印
    printf("Rank %d: MPI Initialized successfully.\n", world_rank);
    fflush(stdout); // 强制刷新缓冲区，确保信息能被立刻看到


    // Original argument parsing - unchanged
    if (argc != 2) {
        if (world_rank == 0) { // Guarded print to avoid multiple outputs
            std::cerr << "Usage: " << argv[0] << " <input_data>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
    std::string filename = argv[1];

    // Original variable declarations - unchanged
    int N;
    double *A = nullptr, *b = nullptr, *x = nullptr;

    // Original data reading block on rank 0 - unchanged
    if (world_rank == 0) {
        printf("Rank 0: Reading data...\n");
        fflush(stdout);
        read_data(filename, &N, &A, &b, &x);
    }

    // --- Start of injected MPI data distribution logic ---

    // 1. Broadcast the problem size N from rank 0 to all other processes
    printf("Rank %d: About to broadcast N.\n", world_rank);
    fflush(stdout);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: Broadcast of N finished. N = %d\n", world_rank, N);
    fflush(stdout);

    // 2. Calculate data distribution for potentially non-divisible N
    int base_load = N / world_size;
    int remainder = N % world_size;
    int local_N = base_load + (world_rank < remainder ? 1 : 0);

    // Prepare counts and displacements for MPI_Scatterv/Gatherv
    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);
    std::vector<int> mat_counts(world_size);
    std::vector<int> mat_displs(world_size);

    int current_displ = 0;
    for (int i = 0; i < world_size; ++i) {
        counts[i] = base_load + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += counts[i];
    }
    for (int i = 0; i < world_size; ++i) {
        mat_counts[i] = counts[i] * N;
        mat_displs[i] = displs[i] * N;
    }

    // 3. Allocate local buffers on ALL processes
    double* local_A = new double[local_N * N];
    double* local_b = new double[local_N];
    double* local_x = new double[local_N];

    // 4. Distribute data from Rank 0 to all processes
    printf("Rank %d: About to scatter data.\n", world_rank);
    fflush(stdout);
    MPI_Scatterv(A, mat_counts.data(), mat_displs.data(), MPI_DOUBLE,
                 local_A, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(b, counts.data(), displs.data(), MPI_DOUBLE,
                 local_b, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(x, counts.data(), displs.data(), MPI_DOUBLE,
                 local_x, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("Rank %d: Scatter finished.\n", world_rank);
    fflush(stdout);
    // --- End of injected MPI data distribution logic ---


    // Original timing block - unchanged
    auto start = std::chrono::high_resolution_clock::now();
    
    // The core solver call is replaced with the MPI version
    int iter = bicgstab_mpi(N, local_N, local_A, local_b, local_x, MAX_ITER, TOL, world_rank, counts.data(), displs.data());
    
    auto end = std::chrono::high_resolution_clock::now();


    // --- Injected MPI result collection logic ---
    // Gather the final result from all processes back into the `x` buffer on rank 0
    MPI_Gatherv(local_x, local_N, MPI_DOUBLE,
                x, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    // --- End of injected MPI result collection logic ---


    // Original result judging block - unchanged
    if (world_rank == 0) {
        auto duration = end - start;
        judge(iter, duration, N, A, b, x);
    }

    // Free local buffers on all processes
    delete[] local_A;
    delete[] local_b;
    delete[] local_x;

    // Original memory cleanup - now guarded, which is a necessary correction for MPI
    // Only rank 0 allocated A, b, and x, so only it should free them.
    if (world_rank == 0) {
        free(A);
        free(b);
        free(x);
    }
    
    // --- MPI logic injected as requested by comments ---
    MPI_Finalize();
    return 0;
}