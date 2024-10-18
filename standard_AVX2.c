// /usr/local/opt/llvm/bin/clang -O3 -mavx2 -mfma -fopenmp -I/usr/local/opt/llvm/include -L/usr/local/opt/llvm/lib -o matrix_mult_avx2 standard_AVX2.c

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

#define NUM 4096  
#define BLOCK_SIZE 128 

void mat_mult_blocked(double *A, double *B_transposed, double *C, int N) {
    int i, j, k, ii, jj, kk;

    #pragma omp parallel for private(i, j, k, ii, jj, kk) schedule(static)
    for (ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                    if (kk + BLOCK_SIZE < N) {
                        _mm_prefetch((const char*)&A[(i + 1) * N + kk], _MM_HINT_T0);
                    }
                    for (j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                        __m256d c_vec = _mm256_setzero_pd();
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k += 4) {
                            if (k + BLOCK_SIZE < N) {
                                _mm_prefetch((const char*)&B_transposed[(j + 1) * N + k], _MM_HINT_T0);
                            }
                            __m256d a_vec = _mm256_load_pd(&A[i * N + k]);
                            __m256d b_vec = _mm256_load_pd(&B_transposed[j * N + k]);
                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                        }
                        double sum = c_vec[0] + c_vec[1] + c_vec[2] + c_vec[3];
                        C[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    double *A = NULL, *B = NULL, *C = NULL;
    double *B_transposed = NULL;

    posix_memalign((void**)&A, 32, NUM * NUM * sizeof(double));
    posix_memalign((void**)&B, 32, NUM * NUM * sizeof(double));
    posix_memalign((void**)&C, 32, NUM * NUM * sizeof(double));
    posix_memalign((void**)&B_transposed, 32, NUM * NUM * sizeof(double));

    srand48(0);  
    for (int i = 0; i < NUM * NUM; i++) {
        A[i] = drand48();
        B[i] = drand48();
        C[i] = 0.0;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM; i++) {
        for (int j = 0; j < NUM; j++) {
            B_transposed[j * NUM + i] = B[i * NUM + j];
        }
    }

    double start_time = omp_get_wtime();
    mat_mult_blocked(A, B_transposed, C, NUM);
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    double gflops = (2.0 * NUM * NUM * NUM) / (total_time * 1e9);
    printf("Exec time : %f s\n", total_time);
    printf("Performances : %f GFLOPS\n", gflops);

    free(A);
    free(B);
    free(C);
    free(B_transposed);

    return 0;
}
