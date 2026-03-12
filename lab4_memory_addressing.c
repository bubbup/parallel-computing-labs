#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>


/* For Windows compatibility with aligned allocation */
#ifdef _WIN32
#include <malloc.h>
#define malloc_aligned(size) _aligned_malloc(size, 64)
#define free_aligned(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
static void* malloc_aligned(size_t size) {
    void* ptr;
    if (posix_memalign(&ptr, 64, size) != 0) return NULL;
    return ptr;
}
#define free_aligned(ptr) free(ptr)
#endif

static double tnow(void) { return omp_get_wtime(); }

/* Initialize array */
static void init_float(float *a, long n) {
    for (long i = 0; i < n; i++) a[i] = 1.0f;
}

/* Task A: Stride Sum */
static float stride_sum(const float *a, long n, int stride) {
    float sum = 0.0f;
    for (long i = 0; i < n; i += stride) {
        sum += a[i];
    }
    return sum;
}

/* Task B: Pointer Chasing */
static long pointer_chase(const int *next, long iters, int start) {
    long idx = (long)start;
    for (long i = 0; i < iters; i++) {
        idx = next[idx];
    }
    return idx;
}

/* Task C: Streaming */
static void stream_copy_omp(const float *a, float *b, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        b[i] = a[i];
    }
}

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

    float *a = (float *)malloc_aligned(n * sizeof(float));
    float *b = (float *)malloc_aligned(n * sizeof(float));
    float *c = (float *)malloc_aligned(n * sizeof(float));
    if (!a || !b || !c) { printf("Alloc failed\n"); return 1; }

    init_float(a, n);
    init_float(b, n);

    /* ---------------- Task A: stride scan ---------------- */
    printf("Task A: stride scan\n");
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
    printf("Task B: pointer chasing\n");
    int *next = (int *)malloc_aligned(n * sizeof(int));
    
    /* Step must be odd to visit all indices if n is power of 2 */
    int step = 104729; 
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
    /* Copy: 1 read + 1 write = 2 floats per element */
    double gbs = (2.0 * sizeof(float) * n) / (t * 1e9); 
    printf(" copy: time=%.6f s BW=%.2f GB/s\n", t, gbs);

    t0 = tnow();
    stream_add_omp(a, b, c, n);
    t = tnow() - t0;
    /* Add: 2 reads + 1 write = 3 floats per element */
    gbs = (3.0 * sizeof(float) * n) / (t * 1e9);
    printf(" add : time=%.6f s BW=%.2f GB/s\n\n", t, gbs);

    free_aligned(a); free_aligned(b); free_aligned(c); free_aligned(next);
    return 0;
}