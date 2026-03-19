#include <stdio.h> 
#include <stdlib.h> 
#include <pthread.h> 
#include <sys/time.h>


typedef struct {
    int tid;
    int nthreads;
    const double *a;
    long N;
    double partial;
} thread_arg_t;

static double tnow(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

static void init_double(double *a, long N) {
    for (long i = 0; i < N; i++) a[i] = 1.0;
}

static void *sum_worker(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    long chunk = t->N / t->nthreads;
    long start = t->tid * chunk;
    // Ensure the last thread takes any remaining elements if N isn't perfectly divisible
    long end = (t->tid == t->nthreads - 1) ? t->N : start + chunk;

    double s = 0.0;
    for (long i = start; i < end; i++) {
        s += t->a[i];
    }
    t->partial = s;
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = (argc >= 2) ? atoi(argv[1]) : 4;
    long N = (argc >= 3) ? atol(argv[2]) : (1L << 24);

    pthread_t *thr = malloc(nthreads * sizeof(pthread_t));
    thread_arg_t *args = malloc(nthreads * sizeof(thread_arg_t));
    double *a = malloc(N * sizeof(double));

    init_double(a, N);

    printf("Summing %ld elements with %d threads...\n", N, nthreads);

    double t_start = tnow();

    /* 1 & 2) Initialize arguments and Create worker threads */
    for (int i = 0; i < nthreads; i++) {
        args[i].tid = i;
        args[i].nthreads = nthreads;
        args[i].a = a;
        args[i].N = N;
        args[i].partial = 0.0;

        if (pthread_create(&thr[i], NULL, sum_worker, &args[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            return 1;
        }
    }

    /* 3) Wait for all threads to finish */
    for (int i = 0; i < nthreads; i++) {
        pthread_join(thr[i], NULL);
    }

    /* 4) Combine partial sums */
    double final_sum = 0.0;
    for (int i = 0; i < nthreads; i++) {
        final_sum += args[i].partial;
    }

    double t_total = tnow() - t_start;

    /* 5) Results and Validation */
    printf("Computed Sum: %.1f\n", final_sum);
    printf("Expected Sum: %.1f\n", (double)N);
    printf("Elapsed Time: %.6f seconds\n", t_total);

    if (final_sum == (double)N) {
        printf("Verification: SUCCESS\n");
    } else {
        printf("Verification: FAILED\n");
    }

    free(a); 
    free(args); 
    free(thr);
    return 0;
}

  
     
       

    