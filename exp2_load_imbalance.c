#include <stdio.h>
#include <omp.h>

// Helper to get time
static double tnow(void) { 
    return omp_get_wtime(); 
}

// Function to simulate work
static void burn(long iters) {
    volatile double x = 0.0;
    for (long k = 0; k < iters; k++) {
        x += k * 1e-12;
    }
}

int main(void) {
    const long N = 200000;
    double t0, t1;

    /* A) Static Scheduling */
    t0 = tnow();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        long cost = (i % 1000 == 0) ? 80000 : 800;
        burn(cost);
    }
    t1 = tnow();
    double TA = t1 - t0;

    /* B) Dynamic Scheduling */
    t0 = tnow();
    // Using a chunk size of 64
    #pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < N; i++) {
        long cost = (i % 1000 == 0) ? 80000 : 800;
        burn(cost);
    }
    t1 = tnow();
    double TB = t1 - t0;

    printf("static:  %.6f seconds\n", TA);
    printf("dynamic: %.6f seconds\n", TB);
    
    return 0;
}