#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>  
#include <omp.h>  

#define N 4096
#define BLOCK_SIZE 64  

void mat_mult_blocked(float *A, float *B_transposed, float *C, int N) {
    int i, j, k, ii, jj, kk;

    #pragma omp parallel for private(i, j, k, ii, jj, kk) schedule(static)
    for (ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                        float32x4_t c_vec = vdupq_n_f32(0.0f);  
                        
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k += 4) {
                            float32x4_t a_vec = vld1q_f32(&A[i * N + k]);  
                            float32x4_t b_vec = vld1q_f32(&B_transposed[j * N + k]); 
                            
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                        }
                        
                        C[i * N + j] += vaddvq_f32(c_vec); 
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    float *A = NULL, *B = NULL, *C = NULL;
    float *B_transposed = NULL;

    posix_memalign((void**)&A, 32, N * N * sizeof(float));
    posix_memalign((void**)&B, 32, N * N * sizeof(float));
    posix_memalign((void**)&C, 32, N * N * sizeof(float));
    posix_memalign((void**)&B_transposed, 32, N * N * sizeof(float));

    srand48(0); 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)drand48();
        B[i] = (float)drand48();
        C[i] = 0.0f;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_transposed[j * N + i] = B[i * N + j];
        }
    }

    double start_time = omp_get_wtime();
    mat_mult_blocked(A, B_transposed, C, N);
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    double gflops = (2.0 * N * N * N) / (total_time * 1e9);
    printf("Exec time : %f secs\n", total_time);
    printf("Performances : %f GFLOPS\n", gflops);
    free(A);
    free(B);
    free(C);
    free(B_transposed);

    return 0;
}
