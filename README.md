# Benchmark-Xeon-Phi-AVX2-AVX512-MPI

for the AVX2 MPI version :


mpicc -O3 -march=native -mfma -mavx2 -ffast-math -ffp-contract=fast -funroll-loops -ftree-vectorize -fopt-info-vec -fopenmp -lnuma -pthread avx2_MPI.c  && export OMP_DYNAMIC=FALSE && export OMP_NESTED=FALSE

numactl -a --interleave=all mpirun -n 36 ./a.out
