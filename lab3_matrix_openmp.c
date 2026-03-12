#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double tnow(void) { return omp_get_wtime(); }

/* Initialize matrices A and B (deterministically) */
void init_mats(int N, float *A, float *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            A[idx] = 1.0f;
            B[idx] = 2.0f;
        }
    }
}

/* TODO #1: Sequential matrix addition: C = A + B */
void mat_add_seq(int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

/* Sequential matrix multiplication: C = A * B */
void mat_mul_seq(int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* TODO #2: OpenMP matrix addition */
void mat_add_omp(int N, const float *A, const float *B, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

/* TODO #3: OpenMP matrix multiplication (naive) */
void mat_mul_omp(int N, const float *A, const float *B, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* TODO #4: Branchy add for coherence/divergence discussion */
void mat_add_branchy_omp(int N, const float *A, const float *B, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            C[idx] = (A[idx] > 0) ? (A[idx] + B[idx]) : (A[idx] - B[idx]);
        }
    }
}

int main(int argc, char **argv) {
    /* TODO #7: read N from argv */
    int N = 512;
    if (argc >= 2) N = atoi(argv[1]);
    size_t NN = (size_t)N * N;

    /* TODO #8: allocate A, B, C */
    float *A = (float *)malloc(NN * sizeof(float));
    float *B = (float *)malloc(NN * sizeof(float));
    float *C = (float *)malloc(NN * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    /* TODO #9: call init_mats */
    init_mats(N, A, B);

    /* TODO #10: warm-up once */
    mat_add_omp(N, A, B, C);
    mat_mul_omp(N, A, B, C);

    int p = omp_get_max_threads();

    /* ---------- ADD timings ---------- */
    double t0 = tnow();
    mat_add_seq(N, A, B, C);
    double T1_add = tnow() - t0;

    t0 = tnow();
    mat_add_omp(N, A, B, C);
    double Tp_add = tnow() - t0;

    /* ---------- MUL timings ---------- */
    t0 = tnow();
    mat_mul_seq(N, A, B, C);
    double T1_mul = tnow() - t0;

    t0 = tnow();
    mat_mul_omp(N, A, B, C);
    double Tp_mul = tnow() - t0;

    /* ---------- Metrics ---------- */
    double S_add = T1_add / Tp_add;
    double E_add = S_add / (double)p;
    double S_mul = T1_mul / Tp_mul;
    double E_mul = S_mul / (double)p;

    /* ---------- Report ---------- */
    printf("N = %d\n", N);
    printf("OpenMP max threads = %d\n\n", p);
    printf("Timings (seconds):\n");
    printf(" add: T1 = %.6f, Tp = %.6f\n", T1_add, Tp_add);
    printf(" mul: T1 = %.6f, Tp = %.6f\n\n", T1_mul, Tp_mul);
    printf("Metrics (baseline = sequential):\n");
    printf(" add: Speedup S(p)=%.4f, Efficiency E(p)=%.4f\n", S_add, E_add);
    printf(" mul: Speedup S(p)=%.4f, Efficiency E(p)=%.4f\n", S_mul, E_mul);

    /* TODO #15: free A, B, C */
    free(A);
    free(B);
    free(C);

    return 0;
}