#include <stdio.h>
#include <omp.h>

static double tnow(void) { 
    return omp_get_wtime(); 
}

int main(void) {
    const long N = 2000000;
    double t0, t1;

    /* A) Fine-grain synchronization */
    // This is EXTREMELY slow because of constant locking/unlocking
    t0 = tnow();
    double sumA = 0.0;
    #pragma omp parallel
    {
        // Each thread runs the loop N times and locks EVERY time
        for (long i = 0; i < N; i++) {
            #pragma omp critical
            sumA += 1.0;
        }
    }
    t1 = tnow();
    double TA = t1 - t0;

    /* B) Per-thread batching */
    // This is fast because work is done locally first
    t0 = tnow();
    double sumB = 0.0;
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for
        for (long i = 0; i < N; i++) {
            local += 1.0;
        }
        #pragma omp atomic
        sumB += local;
    }
    t1 = tnow();
    double TB = t1 - t0;

    printf("fine_sync: %.6f seconds\n", TA);
    printf("batched:   %.6f seconds\n", TB);
    
    return 0;
}