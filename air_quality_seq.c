#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h> // Include OpenMP header

/* Function to get the current time in seconds */
static double now_seconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

/* Synthetic pollution value generator */
static void generate_readings(double *readings, long total) {
    for (long i = 0; i < total; i++) {
        readings[i] = 20.0 + (double)(i % 100) + 5.0 * sin((double)i * 0.001);
    }
}

/* Correction formula */
static double correct_value(double raw) {
    return 1.08 * raw + 2.5;
}

int main(int argc, char **argv) {
    // --- Setup and Memory Allocation ---
    int days = (argc >= 2) ? atoi(argv[1]) : 365;
    int sensors = (argc >= 3) ? atoi(argv[2]) : 100000;
    double threshold = (argc >= 4) ? atof(argv[3]) : 100.0;
    long total = (long)days * sensors;

    double *readings = (double *)malloc(total * sizeof(double));
    if (readings == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    generate_readings(readings, total);

    // --- Sequential Baseline ---
    printf("Running Sequential Baseline...\n");
    double ts0 = now_seconds();
    double s_sum = 0.0;
    double s_max = -DBL_MAX;
    long s_dangerous = 0;

    for (long i = 0; i < total; i++) {
        double corrected = correct_value(readings[i]);
        s_sum += corrected;
        if (corrected > s_max) s_max = corrected;
        if (corrected > threshold) s_dangerous++;
    }
    double ts1 = now_seconds();

    // --- Parallel Implementation (OpenMP) ---
    /* 
       We use 'reduction' to combine local thread results safely.
       OpenMP handles the 'Local Statistics' logic internally:
       - Each thread gets a private copy of sum, max, and count.
       - Threads update their private copies independently.
       - At the end, OpenMP combines them into the global variables.
    */
    printf("Running Parallel (OpenMP)...\n");
    double tp0 = omp_get_wtime(); // More accurate timer for parallel code
    double p_sum = 0.0;
    double p_max = -DBL_MAX;
    long p_dangerous = 0;

    #pragma omp parallel for reduction(+:p_sum, p_dangerous) reduction(max:p_max)
    for (long i = 0; i < total; i++) {
        double corrected = correct_value(readings[i]);
        
        // Addition, Multiplication, and Comparison performed locally by threads
        p_sum += corrected; 
        
        if (corrected > p_max) {
            p_max = corrected;
        }
        
        if (corrected > threshold) {
            p_dangerous++;
        }
    }
    double tp1 = omp_get_wtime();

    // --- Output Results ---
    printf("\n--- Results ---\n");
    printf("Days = %d | Sensors = %d | Total = %ld\n", days, sensors, total);
    printf("Threshold = %.2f\n", threshold);
    
    printf("\n[Sequential]\n");
    printf("Average: %.6f | Max: %.6f | Dangerous: %ld\n", s_sum/total, s_max, s_dangerous);
    printf("Runtime: %.6f seconds\n", ts1 - ts0);

    printf("\n[Parallel OpenMP]\n");
    printf("Average: %.6f | Max: %.6f | Dangerous: %ld\n", p_sum/total, p_max, p_dangerous);
    printf("Runtime: %.6f seconds\n", tp1 - tp0);

    printf("\nSpeedup: %.2fx\n", (ts1 - ts0) / (tp1 - tp0));

    free(readings);
    return 0;
}