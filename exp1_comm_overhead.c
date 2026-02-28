#include <stdio.h>
#include <omp.h>

// Helper function to get current time
static double tnow(void) { 
    return omp_get_wtime(); 
}

int main(void) {
    const long N = 20000000; 
    double t0, t1;

    /* A) Critical combine */
    double sumA = 0.0;
    t0 = tnow();
    
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for
        for(long i = 0; i < N; i++) {
            local += 1.0;
        }
        
        #pragma omp critical
        {
            sumA += local;
        }
    }
    t1 = tnow();
    double TA = t1 - t0;

    /* B) Reduction combine */
    double sumB = 0.0;
    t0 = tnow();
    
    #pragma omp parallel for reduction(+:sumB)
    for(long i = 0; i < N; i++) {
        sumB += 1.0;
    }
    
    t1 = tnow();
    double TB = t1 - t0;

    printf("critical %.6f reduction %.6f\n", TA, TB);
    
    return 0;
}