// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// void gemv(double* __restrict y, double* __restrict A, double* __restrict x, int N) {
//     // y = A * x
//     #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         y[i] = 0.0;
//         for (int j = 0; j < N; j++) {
//             y[i] += A[i * N + j] * x[j];
//         }
//     }
// }

// double dot_product(double* __restrict x, double* __restrict y, int N) {
//     // dot product of x and y
//     double result = 0.0;
//     #pragma omp parallel for simd reduction(+:result)
//     for (int i = 0; i < N; i++) {
//         result += x[i] * y[i];
//     }
//     return result;
// }

// void precondition(double* __restrict A, double* __restrict K2_inv, int N) {
//     // K2_inv = 1 / diag(A)
//     #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         K2_inv[i] = 1.0 / A[i * N + i];
//     }
// }

// void precondition_apply(double* __restrict z, double* __restrict K2_inv, double* __restrict r, int N) {
//     // z = K2_inv * r
//     #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         z[i] = K2_inv[i] * r[i];
//     }
// }

// int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol) {
//     /**
//      * Algorithm: BICGSTAB
//      *  r: residual
//      *  r_hat: modified residual
//      *  p: search direction
//      *  K2_inv: preconditioner (We only store the diagonal of K2_inv)
//      * Reference: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
//      */
//     double* r      = (double*)calloc(N, sizeof(double));
//     double* r_hat  = (double*)calloc(N, sizeof(double));
//     double* p      = (double*)calloc(N, sizeof(double));
//     double* v      = (double*)calloc(N, sizeof(double));
//     double* s      = (double*)calloc(N, sizeof(double));
//     double* h      = (double*)calloc(N, sizeof(double));
//     double* t      = (double*)calloc(N, sizeof(double));
//     double* y      = (double*)calloc(N, sizeof(double));
//     double* z      = (double*)calloc(N, sizeof(double));
//     double* K2_inv = (double*)calloc(N, sizeof(double));

//     double rho_old = 1, alpha = 1, omega = 1;
//     double rho = 1, beta = 1;
//     double tol_squared = tol * tol;

//     // Take M_inv as the preconditioner
//     // Note that we only use K2_inv (in wikipedia)
//     precondition(A, K2_inv, N);

//     // 1. r0 = b - A * x0
//     gemv(r, A, x, N);
//     #pragma omp parallel for
//     for (int i = 0; i < N; i++) {
//         r[i] = b[i] - r[i];
//     }

//     // 2. Choose an arbitary vector r_hat that is not orthogonal to r
//     // We just take r_hat = r, please do not change this initial value
//     memmove(r_hat, r, N * sizeof(double));  // memmove is safer memcpy :)

//     // 3. rho_0 = (r_hat, r)
//     rho = dot_product(r_hat, r, N);

//     // 4. p_0 = r_0
//     memmove(p, r, N * sizeof(double));

//     int iter;
//     for (iter = 1; iter <= max_iter; iter++) {
//         if (iter % 1000 == 0) {
//             printf("Iteration %d, residul = %e\n", iter, sqrt(dot_product(r, r, N)));
//         }

//         // 1. y = K2_inv * p (apply preconditioner)
//         precondition_apply(y, K2_inv, p, N);

//         // 2. v = Ay
//         gemv(v, A, y, N);

//         // 3. alpha = rho / (r_hat, v)
//         alpha = rho / dot_product(r_hat, v, N);

//         // 4. h = x_{i-1} + alpha * y
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             h[i] = x[i] + alpha * y[i];
//         }

//         // 5. s = r_{i-1} - alpha * v
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             s[i] = r[i] - alpha * v[i];
//         }

//         // 6. Is h is accurate enough, then x_i = h and quit
//         if (dot_product(s, s, N) < tol_squared) {
//             memmove(x, h, N * sizeof(double));
//             break;
//         }

//         // 7. z = K2_inv * s
//         precondition_apply(z, K2_inv, s, N);

//         // 8. t = Az
//         gemv(t, A, z, N);

//         // 9. omega = (t, s) / (t, t)
//         omega = dot_product(t, s, N) / dot_product(t, t, N);

//         // 10. x_i = h + omega * z
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             x[i] = h[i] + omega * z[i];
//         }

//         // 11. r_i = s - omega * t
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             r[i] = s[i] - omega * t[i];
//         }

//         // 12. If x_i is accurate enough, then quit
//         if (dot_product(r, r, N) < tol_squared) break;

//         rho_old = rho;
//         // 13. rho_i = (r_hat, r)
//         rho = dot_product(r_hat, r, N);

//         // 14. beta = (rho_i / rho_{i-1}) * (alpha / omega)
//         beta = (rho / rho_old) * (alpha / omega);

//         // 15. p_i = r_i + beta * (p_{i-1} - omega * v)
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             p[i] = r[i] + beta * (p[i] - omega * v[i]);
//         }
//     }

//     free(r);
//     free(r_hat);
//     free(p);
//     free(v);
//     free(s);
//     free(h);
//     free(t);
//     free(y);
//     free(z);
//     free(K2_inv);

//     if (iter >= max_iter)
//         return -1;
//     else
//         return iter;
// }
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/**
 * @brief Hybrid (MPI+OpenMP) Matrix-Vector Multiplication.
 * y_local = A_local * x_global
 */
// void gemv_hybrid(double* local_y, double* local_A, double* local_x,
//                  int local_N, int N,
//                  int* counts, int* displs) {
//     // 1. 在函数内部管理临时全局向量，使用 static 避免重复分配。
//     //    注意：这种方法假定 N 的值在程序运行期间不变。
//     static double* x_global = NULL;
//     if (x_global == NULL) {
//         x_global = (double*)malloc(N * sizeof(double));
//     }

//     // 2. 所有进程将各自的 local_x 数据块收集到临时的 x_global 缓冲区中。
//     //    这里的第一个 local_x 是传入的输入向量分块。
//     MPI_Allgatherv(local_x, local_N, MPI_DOUBLE,
//                    x_global, counts, displs, MPI_DOUBLE,
//                    MPI_COMM_WORLD);

//     // 3. 每个进程使用完整的 x_global 向量和自己的 local_A 矩阵块进行计算。
//     #pragma omp parallel for
//     for (int i = 0; i < local_N; i++) {
//         local_y[i] = 0.0;
//         for (int j = 0; j < N; j++) {
//             local_y[i] += local_A[i * N + j] * x_global[j];
//         }
//     }
// }
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
int bicgstab_mpi(int N, int local_N, double* local_A, double* local_b, double* local_x,
                 int max_iter, double tol, int rank, int* counts, int* displs) {
    
    // --- Allocate local vectors for this process ---
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
    
    // Global vectors are needed as receive buffers for communication
    // double* y_global      = (double*)malloc(N * sizeof(double));
    // double* z_global      = (double*)malloc(N * sizeof(double));

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho, beta;
    double tol_squared = tol * tol;

    precondition_local(local_A, local_K2_inv, local_N, N, displs[rank]);

    // 1. r0 = b - A * x0
    gemv_hybrid(local_r, local_A, local_x, local_N, N, rank,counts, displs); // y_global is used as a temp buffer
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
            // Then, only rank 0 prints the result. This avoids deadlock.
            if (rank == 0) {
                printf("Iteration %d, residual = %e\n", iter, sqrt(residual_norm_sq));
                fflush(stdout);
            }
        }

        // 1. y = K2_inv * p (apply preconditioner)
        precondition_apply_local(local_y, local_K2_inv, local_p, local_N);

        // 2. v = Ay
        gemv_hybrid(local_v, local_A, local_y, local_N, N, rank,counts, displs);

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
        gemv_hybrid(local_t, local_A, local_z, local_N, N, rank,counts, displs);

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
        }

        // 11. r_i = s - omega * t
        #pragma omp parallel for
        for (int i = 0; i < local_N; i++) {
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
    // free(y_global); free(z_global);

    if (iter > max_iter) return -1;
    return iter;
}