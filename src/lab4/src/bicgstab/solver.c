#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

void gemv_hybrid(double* local_y, double* local_A, double* local_x,
                       int local_N, int N, int rank,
                       int* counts, int* displs) {
    // Use static buffers for the global vector and MPI request handle.
    static double* x_global = NULL;
    static MPI_Request request = MPI_REQUEST_NULL;
    if (x_global == NULL) {
        x_global = (double*)malloc(N * sizeof(double));
    }

    // 1. Initiate non-blocking communication (MPI_Iallgatherv).
    //    This starts gathering data from all processes into x_global in the background.
    MPI_Iallgatherv(local_x, local_N, MPI_DOUBLE,
                    x_global, counts, displs, MPI_DOUBLE,
                    MPI_COMM_WORLD, &request);

    // 2. Overlap: Compute the "internal" part of the multiplication.
    //    This part only depends on local_x, which this process already has.
    //    We do this while the data from other processes is being transferred.
    int local_start_col = displs[rank];
    int local_num_cols = counts[rank];

    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) {
        double partial_sum = 0.0;
        // The columns from local_start_col to local_start_col + local_num_cols
        // correspond to the data in local_x.
        #pragma omp simd reduction(+:partial_sum)
        for (int j = 0; j < local_num_cols; j++) {
            partial_sum += local_A[i * N + (local_start_col + j)] * local_x[j];
        }
        local_y[i] = partial_sum;
    }

    // 3. Wait for the non-blocking communication to complete.
    //    By this point, a significant portion of computation is already done.
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // 4. Compute the "external" part of the multiplication.
    //    This uses the data in x_global that has just arrived from other processes.
    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) {
        // Use a temporary scalar to remove data dependency for the vectorizer.
        double external_sum = 0.0;

        // Part 1: Columns before the local block
        #pragma omp simd reduction(+:external_sum)
        for (int j = 0; j < local_start_col; j++) {
            external_sum += local_A[i * N + j] * x_global[j];
        }

        // Part 2: Columns after the local block
        #pragma omp simd reduction(+:external_sum)
        for (int j = local_start_col + local_num_cols; j < N; j++) {
            external_sum += local_A[i * N + j] * x_global[j];
        }

        // Final update after the inner loops are complete.
        local_y[i] += external_sum;
    }
}

/**
 * @brief Hybrid (MPI+OpenMP) Dot Product.
 */
double dot_product_hybrid(double* local_x, double* local_y, int local_N) {
    double local_sum = 0.0;
    // Each process computes a partial dot product using its local data with OpenMP.
    #pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < local_N; i++) {
        local_sum += local_x[i] * local_y[i];
    }

    double global_sum = 0.0;
    // Sum up the partial results from all processes.
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

/**
 * @brief Preconditioner setup (runs locally on each process).
 */
void precondition_local(double* local_A, double* local_K2_inv, int local_N, int N, int start_row) {
    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) {
        int global_row = start_row + i;
        local_K2_inv[i] = 1.0 / local_A[i * N + global_row];
    }
}

/**
 * @brief Apply preconditioner (runs locally on each process).
 */
void precondition_apply_local(double* local_z, double* local_K2_inv, double* local_r, int local_N) {
    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) {
        local_z[i] = local_K2_inv[i] * local_r[i];
    }
}

/**
 * @brief The main BiCGSTAB solver, parallelized with MPI and OpenMP.
 */
int bicgstab(int N, int world_size,int world_rank,double* A, double* b, double* x, int max_iter, double tol)
{
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Calculate data distribution for potentially non-divisible N
    int base_load = N / world_size;
    int remainder = N % world_size;
    int local_N = base_load + (world_rank < remainder ? 1 : 0);

    // Prepare counts and displacements for MPI_Scatterv/Gatherv
    int* counts = (int*)malloc(world_size * sizeof(int));
    int* displs = (int*)malloc(world_size * sizeof(int));
    int* mat_counts = (int*)malloc(world_size * sizeof(int));
    int* mat_displs = (int*)malloc(world_size * sizeof(int));

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

    // Allocate local buffers on ALL processes
    double* local_A = (double*)malloc(local_N * N * sizeof(double));
    double* local_b = (double*)malloc(local_N * sizeof(double));
    double* local_x = (double*)malloc(local_N * sizeof(double));

    // Distribute data from Rank 0 to all processes
    MPI_Scatterv(A, mat_counts, mat_displs, MPI_DOUBLE,
                 local_A, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(b, counts, displs, MPI_DOUBLE,
                 local_b, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(x, counts, displs, MPI_DOUBLE,
                 local_x, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    double* local_r       = (double*)malloc(local_N * sizeof(double));
    double* local_r_hat   = (double*)malloc(local_N * sizeof(double));
    double* local_p       = (double*)malloc(local_N * sizeof(double));
    double* local_v       = (double*)malloc(local_N * sizeof(double));
    double* local_s       = (double*)malloc(local_N * sizeof(double));
    double* local_h       = (double*)malloc(local_N * sizeof(double));
    double* local_t       = (double*)malloc(local_N * sizeof(double));
    double* local_y       = (double*)malloc(local_N * sizeof(double));
    double* local_z       = (double*)malloc(local_N * sizeof(double));
    double* local_K2_inv  = (double*)malloc(local_N * sizeof(double));

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho, beta;
    double tol_squared = tol * tol;

    precondition_local(local_A, local_K2_inv, local_N, N, displs[world_rank]);

    // 1. r0 = b - A * x0
    gemv_hybrid(local_r, local_A, local_x, local_N, N, world_rank,counts, displs); // y_global is used as a temp buffer
    #pragma omp parallel for
    for (int i = 0; i < local_N; i++) {
        local_r[i] = local_b[i] - local_r[i];
    }

    // 2. r_hat = r
    memcpy(local_r_hat, local_r, local_N * sizeof(double));

    // 3. rho_0 = (r_hat, r)
    rho = dot_product_hybrid(local_r_hat, local_r, local_N);

    // 4. p_0 = r_0
    memcpy(local_p, local_r, local_N * sizeof(double));

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        if (iter % 1000 == 0) {
            double residual_norm_sq = dot_product_hybrid(local_r, local_r, local_N);
            if (world_rank == 0) {
                printf("Iteration %d, residual = %e\n", iter, sqrt(residual_norm_sq));
                fflush(stdout);
            }
        }

        // 1. y = K2_inv * p (apply preconditioner)
        precondition_apply_local(local_y, local_K2_inv, local_p, local_N);

        // 2. v = Ay
        gemv_hybrid(local_v, local_A, local_y, local_N, N, world_rank,counts, displs);

        // 3. alpha = rho / (r_hat, v)
        alpha = rho / dot_product_hybrid(local_r_hat, local_v, local_N);

        // 4. h = x_{i-1} + alpha * y
        // 5. s = r_{i-1} - alpha * v
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) {
            local_h[i] = local_x[i] + alpha * local_y[i];
            local_s[i] = local_r[i] - alpha * local_v[i];
        }

        // 6. Early exit check
        if (dot_product_hybrid(local_s, local_s, local_N) < tol_squared) {
            memcpy(local_x, local_h, local_N * sizeof(double));
            break;
        }

        // 7. z = K2_inv * s
        precondition_apply_local(local_z, local_K2_inv, local_s, local_N);

        // 8. t = Az
        gemv_hybrid(local_t, local_A, local_z, local_N, N, world_rank,counts, displs);

        // 9. omega = (t, s) / (t, t)
        // omega = dot_product_hybrid(local_t, local_s, local_N) / dot_product_hybrid(local_t, local_t, local_N);
        // Use a single Allreduce to get both global sums.
        double ts_sum = 0.0;
        double tt_sum = 0.0;
        #pragma omp parallel for reduction(+:ts_sum, tt_sum)
        for (int i = 0; i < local_N; i++) {
            ts_sum += local_t[i] * local_s[i];
            tt_sum += local_t[i] * local_t[i];
        }
        
        // This is the crucial step you correctly identified was missing.
        double local_sums[2] = {ts_sum, tt_sum};

        // 2. Use a single Allreduce to get both global sums.
        double global_sums[2];
        MPI_Allreduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        omega = global_sums[0] / global_sums[1];

        // 10. x_i = h + omega * z
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) {
            local_x[i] = local_h[i] + omega * local_z[i];
            local_r[i] = local_s[i] - omega * local_t[i];
        }

        // 12. Final convergence check
        if (dot_product_hybrid(local_r, local_r, local_N) < tol_squared) break;

        rho_old = rho;
        // 13. rho_i = (r_hat, r)
        rho = dot_product_hybrid(local_r_hat, local_r, local_N);

        // 14. beta = (rho_i / rho_{i-1}) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // 15. p_i = r_i + beta * (p_{i-1} - omega * v)
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) {
            local_p[i] = local_r[i] + beta * (local_p[i] - omega * local_v[i]);
        }
    }

    free(local_r); free(local_r_hat); free(local_p); free(local_v);
    free(local_s); free(local_h); free(local_t); free(local_y);
    free(local_z); free(local_K2_inv);
    MPI_Gatherv(local_x, local_N, MPI_DOUBLE,
            x, counts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    free(local_A);
    free(local_b);
    free(local_x);
    free(counts);
    free(displs);
    free(mat_counts);
    free(mat_displs);
    if (iter > max_iter) return -1;
    return iter;
}