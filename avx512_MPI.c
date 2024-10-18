#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  
#include <omp.h>  
#include <memkind.h>   

#define N 4096
#define BLOCK_SIZE 256  

void mat_mult_blocked(double *A_local, double *B_transposed, double *C_local, int n_local, int N_global) {
    int i, j, k, ii, jj, kk;

    #pragma omp parallel for private(i, j, k, ii, jj, kk) schedule(static)
    for (ii = 0; ii < n_local; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N_global; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N_global; kk += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < n_local; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < N_global; j++) {
                        __m512d c_vec = _mm512_setzero_pd();
                        for (k = kk; k < kk + BLOCK_SIZE && k < N_global; k += 8) {
                            __m512d a_vec = _mm512_load_pd(&A_local[i * N_global + k]);
                            __m512d b_vec = _mm512_load_pd(&B_transposed[j * N_global + k]);
                            c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
                        }
                        double sum = _mm512_reduce_add_pd(c_vec);
                        C_local[i * N_global + j] += sum;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    struct memkind *mcdram_kind = MEMKIND_DEFAULT;

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

    int n_local = N / size;

    double *A_local = (double*)memkind_malloc(MEMKIND_HIGH_BANDWIDTH, n_local * N * sizeof(double));
    double *B = (double*)memkind_malloc(MEMKIND_HIGH_BANDWIDTH, N * N * sizeof(double));
    double *C_local = (double*)memkind_malloc(MEMKIND_HIGH_BANDWIDTH, n_local * N * sizeof(double));
    double *B_transposed = (double*)memkind_malloc(MEMKIND_HIGH_BANDWIDTH, N * N * sizeof(double));

    if (!A_local || !B || !C_local || !B_transposed) {
        fprintf(stderr, "Échec de l'allocation mémoire sur la MCDRAM.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

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
        C = (double*)memkind_malloc(MEMKIND_HIGH_BANDWIDTH, N * N * sizeof(double));
    }
    MPI_Gather(C_local, n_local * N, MPI_DOUBLE, C, n_local * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double total_time = end_time - start_time;
    if (rank == 0) {
        printf("Exec time : %f secs\n", total_time);
        double gflops = (2.0 * N * N * N) / (total_time * 1e9);
        printf("Performances : %f GFLOPS\n", gflops);
    }

    memkind_free(MEMKIND_HIGH_BANDWIDTH, A_local);
    memkind_free(MEMKIND_HIGH_BANDWIDTH, B);
    memkind_free(MEMKIND_HIGH_BANDWIDTH, B_transposed);
    memkind_free(MEMKIND_HIGH_BANDWIDTH, C_local);
    if (rank == 0) memkind_free(MEMKIND_HIGH_BANDWIDTH, C);

    MPI_Finalize();
    return 0;
}
