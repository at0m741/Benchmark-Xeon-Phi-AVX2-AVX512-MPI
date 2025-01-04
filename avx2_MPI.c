#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <numa.h>
#include <sys/types.h>
#include <pthread.h>

//BLOCK SIZE 3888 because of my dual Xeon E5-2699v3 that have 36 core each so 72 threads. 
//and for an optimal MPI repartition I need this size for teh benchmark

#define N 3888
#define BLOCK_SIZE 64  /*optimal block size to fit into cache*/
static inline void mat_mult_blocked(double *A_local, double *B_transposed, double *C_local, int n_local, int N_global) {
    int i, j, k, ii, jj, kk;

    #pragma omp parallel for collapse(2) private(i, j, k, ii, jj, kk) schedule(dynamic, 4)
    for (ii = 0; ii < n_local; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N_global; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N_global; kk += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < n_local; i++) {
                    _mm_prefetch((const char*)&A_local[(i + 1) * N_global + kk], _MM_HINT_T0);
                    for (j = jj; j < jj + BLOCK_SIZE && j < N_global; j++) {
                        __m256d c_vec = _mm256_setzero_pd();
                        for (k = kk; k < kk + BLOCK_SIZE && k < N_global; k += 4) {
                            _mm_prefetch((const char*)&B_transposed[(j + 1) * N_global + k], _MM_HINT_T0);
                            __m256d a_vec = _mm256_load_pd(&A_local[i * N_global + k]);
                            __m256d b_vec = _mm256_load_pd(&B_transposed[j * N_global + k]);
                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                        }
                        C_local[i * N_global + j] += c_vec[0] + c_vec[1] + c_vec[2] + c_vec[3];
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
        if (size == 72)
                omp_set_num_threads(36);
        else
                omp_set_num_threads(36);
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Size of matrix N must be divisible by number of MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

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

    MPI_Bcast(B_transposed, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    mat_mult_blocked(A_local, B_transposed, C_local, n_local, N);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    double *C = NULL;
    if (rank == 0) {
        C = (double*) numa_alloc_onnode(N * N * sizeof(double), rank % 2);
    }
    MPI_Gather(C_local, n_local * N, MPI_DOUBLE, C, n_local * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double total_time = end_time - start_time;
    if (rank == 0) {
        printf("Exec time : %f secs\n", total_time);
        double gflops = (2.0 * N * N * N) / (total_time * 1e9);
        printf("Performances : %f GFLOPS\n", gflops);
    }

    numa_free(A_local, n_local * N * sizeof(double));
    numa_free(B, N * N * sizeof(double));
    numa_free(B_transposed, N * N * sizeof(double));
    numa_free(C_local, n_local * N * sizeof(double));
    if (rank == 0) numa_free(C, N * N * sizeof(double));

    MPI_Finalize();
    return 0;
}
