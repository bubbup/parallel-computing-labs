#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>


static double tnow(void) { return omp_get_wtime(); }

static void *xmalloc_aligned(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return p;
}

static void init_float(float *a, long n) {
    for (long i = 0; i < n; i++) a[i] = 1.0f;
}

/* TODO A1: Sum array elements with a configurable stride. */
static float stride_sum(const float *a, long n, int stride) {
    float sum = 0.0f;
    for (long i = 0; i < n; i += stride) {
        sum += a[i];
    }
    return sum;
}

/* TODO B1: Pointer chasing (dependent load chain). */
static long pointer_chase(const int *next, long iters, int start) {
    long idx = (long)start;
    for (long i = 0; i < iters; i++) {
        idx = next[idx];
    }
    return idx;
}

/* TODO C1: OpenMP streaming copy: b[i] = a[i] */
static void stream_copy_omp(const float *a, float *b, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        b[i] = a[i];
    }
}

/* TODO C2: OpenMP streaming add: c[i] = a[i] + b[i] */
static void stream_add_omp(const float *a, const float *b, float *c, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    long n = (argc >= 2) ? atol(argv[1]) : (1L << 26); 
    int iters = (argc >= 3) ? atoi(argv[2]) : 200000000; 
    
    printf("n = %ld floats (%.2f MB)\n", n, (n * sizeof(float)) / (1024.0*1024.0));
    printf("OpenMP max threads = %d\n\n", omp_get_max_threads());

    float *a = (float *)xmalloc_aligned(n * sizeof(float));
    float *b = (float *)xmalloc_aligned(n * sizeof(float));
    float *c = (float *)xmalloc_aligned(n * sizeof(float));
    init_float(a, n);
    init_float(b, n);

    /* ---------------- Task A: stride scan ---------------- */
    printf("Task A: stride scan (time per accessed element)\n");
    for (int stride = 1; stride <= 128; stride *= 2) {
        double t0 = tnow();
        float s = stride_sum(a, n, stride);
        double t = tnow() - t0;
        long accesses = (n + stride - 1) / stride;
        double ns_per = 1e9 * t / (double)accesses;
        printf(" stride=%3d accesses=%ld time=%.6f s %.2f ns/access (sum=%g)\n",
               stride, accesses, t, ns_per, s);
    }
    printf("\n");

    /* ---------------- Task B: pointer chase ---------------- */
    printf("Task B: pointer chasing (latency bound)\n");
    int *next = (int *)xmalloc_aligned(n * sizeof(int));
    
    /* TODO B2: Build a simple permutation cycle. 
       Using an odd step ensures we hit every index if n is a power of 2. */
    int step = 104729; // A large prime to jump across cache lines
    for (long i = 0; i < n; i++) {
        next[i] = (int)((i + step) % n);
    }

    double t0 = tnow();
    long last = pointer_chase(next, (long)iters, 0);
    double t = tnow() - t0;
    double ns_per = 1e9 * t / (double)iters;
    printf(" iters=%d time=%.6f s %.2f ns/iter (last=%ld)\n\n", iters, t, ns_per, last);

    /* ---------------- Task C: bandwidth stream ---------------- */
    printf("Task C: streaming bandwidth\n");
    stream_copy_omp(a, c, n); // warmup
    
    t0 = tnow();
    stream_copy_omp(a, c, n);
    t = tnow() - t0;
    // Copy reads 4 bytes and writes 4 bytes per element (8 bytes total)
    double gbs = (8.0 * (double)n * sizeof(float)) / (t * 1e9); 
    printf(" copy: time=%.6f s BW=%.2f GB/s\n", t, gbs);

    t0 = tnow();
    stream_add_omp(a, b, c, n);
    t = tnow() - t0;
    // Add reads a, reads b, and writes c (12 bytes total per element)
    gbs = (12.0 * (double)n * sizeof(float)) / (t * 1e9);
    printf(" add : time=%.6f s BW=%.2f GB/s\n\n", t, gbs);

    free(a); free(b); free(c); free(next);
    return 0;
}