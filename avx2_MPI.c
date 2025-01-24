#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <numa.h>
#include <sys/types.h>
#include <pthread.h>
#include <cpuid.h>

#define N 16384
#define BLOCK_SIZE 128

static inline void mat_mult_blocked(double *A_local, double *B_transposed, double *C_local, int n_local, int N_global) {
    int i, j, k, ii, jj, kk;


    #pragma omp parallel for simd collapse(2) private(i, j, k, ii, jj, kk) schedule(guided, BLOCK_SIZE / 4)
    for (ii = 0; ii < n_local; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N_global; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N_global; kk += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < n_local; i++) {
                    _mm_prefetch((const char*)&A_local[(i + 1) * N_global + kk], _MM_HINT_T0);
                    for (j = jj; j < jj + BLOCK_SIZE && j < N_global; j++) {
                        // Charger C_local existant AVANT d'accumuler
                        _mm256_storeu_pd(&C_local[i * N_global + j],
                            _mm256_fmadd_pd(
                                _mm256_load_pd(&A_local[i * N_global + kk]),
                                _mm256_load_pd(&B_transposed[j * N_global + kk]),
                                _mm256_loadu_pd(&C_local[i * N_global + j]) // Charger C_local actuel
                            )
                        );
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Size of matrix N must be divisible by number of MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    omp_set_num_threads(36);
    int n_local = N / size;
    numa_run_on_node(rank % 2);
    numa_set_preferred(rank % 2);
    double *A_local = (double*) numa_alloc_onnode(n_local * N * sizeof(double), rank % 2);
    double *B = (double*) numa_alloc_onnode(N * N * sizeof(double), rank % 2);
    double *C_local = (double*) numa_alloc_onnode(n_local * N * sizeof(double), rank % 2);
    double *B_transposed = (double*) numa_alloc_onnode(N * N * sizeof(double), rank % 2);
    srand48(rank);
    for (int i = 0; i < n_local * N; i++) A_local[i] = drand48();
    for (int i = 0; i < N * N; i++) B[i] = drand48();
    for (int i = 0; i < n_local * N; i++) C_local[i] = 0.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_transposed[j * N + i] = B[i * N + j];
        }
    }
    double mpi_bcast_start = MPI_Wtime();
    MPI_Bcast(B_transposed, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double mpi_bcast_end = MPI_Wtime();
    if (rank == 0) printf("MPI_Bcast Time: %f sec\n", mpi_bcast_end - mpi_bcast_start);
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    mat_mult_blocked(A_local, B_transposed, C_local, n_local, N);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double global_time;
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double global_gflops = ((2.0 * n_local * N * N) / (global_time * 1e9));
    if (rank == 0) {
        printf("Global Performances : %f GFLOPS\n", global_gflops);
    }
    numa_free(A_local, n_local * N * sizeof(double));
    numa_free(B, N * N * sizeof(double));
    numa_free(B_transposed, N * N * sizeof(double));
    numa_free(C_local, n_local * N * sizeof(double));
    MPI_Finalize();
    return 0;
}

