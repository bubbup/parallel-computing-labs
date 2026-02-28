#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double tnow(void) {
    return omp_get_wtime();
}

/* TODO #1: Sequential summation (baseline) */
double sum_seq(const double *a, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

/* TODO #2: Parallel summation using critical */
double sum_parallel_critical(const double *a, long N) {
    double total_sum = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        // Divide the loop iterations among threads
        #pragma omp for
        for (long i = 0; i < N; i++) {
            local_sum += a[i];
        }
        // Thread-safe update to the global sum
        #pragma omp critical
        {
            total_sum += local_sum;
        }
    }
    return total_sum;
}

/* TODO #3: Parallel summation using reduction */
double sum_parallel_reduction(const double *a, long N) {
    double sum = 0.0;
    // OpenMP handles local sums and combining them automatically
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        sum += a[i];
    }
    return sum;
}

int main(void) {
    const long N = 1000000L; 
    double *a;
    double t0, t1;
    double Tseq, Tcrit, Tred;
    double s_seq, s_crit, s_red;

    a = (double *)malloc((size_t)N * sizeof(double));
    if (a == NULL) {
        fprintf(stderr, "Error: allocation failed for N=%ld\n", N);
        return 1;
    }

    /* TODO #4: Initialize the array values */
    for (long i = 0; i < N; i++) {
        a[i] = 1.0; // Simple initialization for testing
    }

    /* Baseline (sequential) */
    t0 = tnow();
    s_seq = sum_seq(a, N);
    t1 = tnow();
    Tseq = t1 - t0;

    /* Parallel (critical) */
    t0 = tnow();
    s_crit = sum_parallel_critical(a, N);
    t1 = tnow();
    Tcrit = t1 - t0;

    /* Parallel (reduction) */
    t0 = tnow();
    s_red = sum_parallel_reduction(a, N);
    t1 = tnow();
    Tred = t1 - t0;

    /* Correctness check */
    const double eps = 1e-6;
    /* TODO #5: Compare s_seq with s_crit and s_red using eps */
    int ok_crit = (fabs(s_seq - s_crit) < eps);
    int ok_red = (fabs(s_seq - s_red) < eps);

    /* Metrics */
    int p = omp_get_max_threads();

    /* TODO #6 & #7: Compute speedup and efficiency */
    // Using equations: S(p) = T(1)/T(p) and E(p) = S(p)/p
    double Scrit = Tseq / Tcrit;
    double Ecrit = Scrit / p;

    double Sred = Tseq / Tred;
    double Ered = Sred / p;

    /* Report */
    printf("N = %ld\n", N);
    printf("OpenMP max threads = %d\n\n", p);
    printf("Results (sum values):\n");
    printf("  seq       = %.6f\n", s_seq);
    printf("  critical  = %.6f [%s]\n", s_crit, ok_crit ? "OK" : "Mismatch");
    printf("  reduction = %.6f [%s]\n\n", s_red, ok_red ? "OK" : "Mismatch");

    printf("Timings (seconds):\n");
    printf("  T_seq       = %.6f\n", Tseq);
    printf("  T_critical  = %.6f\n", Tcrit);
    printf("  T_reduction = %.6f\n\n", Tred);

    printf("Metrics (using T_seq as baseline):\n");
    printf("  critical : S(p)=%.4f, E(p)=%.4f\n", Scrit, Ecrit);
    printf("  reduction: S(p)=%.4f, E(p)=%.4f\n", Sred, Ered);

    free(a);
    return 0;
}