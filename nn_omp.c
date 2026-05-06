#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <omp.h> // TODO 1: Include OpenMP

#define IDX2(i, j, cols) ((i) * (cols) + (j))

// TODO 1: Enable OpenMP timing
static double now_seconds(void) {
    return omp_get_wtime();
}

static double rand_small(int i) {
    return sin((double)i * 0.013) * 0.01;
}

static int min_int(int a, int b) {
    return a < b ? a : b;
}

static void generate_data(double *X, int *y, int N, int D, int C) {
    for (int i = 0; i < N; i++) {
        y[i] = i % C;
        for (int j = 0; j < D; j++) {
            X[IDX2(i, j, D)] = sin(0.01 * i + 0.02 * j) + 0.1 * (double)y[i];
        }
    }
}

static void initialize_weights(double *W, int size) {
    for (int i = 0; i < size; i++) {
        W[i] = rand_small(i);
    }
}

// Parallel accuracy computation
static double compute_accuracy(const double *X, const int *y, const double *W1, const double *W2, const double *b1, const double *b2, int N, int D, int H, int C) {
    int correct = 0;
    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int i = 0; i < N; i++) {
        double A1_local[H]; 
        for (int h = 0; h < H; h++) {
            double sum = b1[h];
            for (int d = 0; d < D; d++) sum += X[IDX2(i, d, D)] * W1[IDX2(d, h, H)];
            A1_local[h] = sum > 0.0 ? sum : 0.0;
        }
        int best_class = 0;
        double best_score = -DBL_MAX;
        for (int c = 0; c < C; c++) {
            double sum = b2[c];
            for (int h = 0; h < H; h++) sum += A1_local[h] * W2[IDX2(h, c, C)];
            if (sum > best_score) { best_score = sum; best_class = c; }
        }
        if (best_class == y[i]) correct++;
    }
    return (double)correct / (double)N;
}

int main(int argc, char **argv) {
    // TODO 2: Add runtime scheduling arguments
    if (argc != 9) {
        fprintf(stderr, "Usage: %s N D H C B epochs schedule chunk\n", argv[0]);
        fprintf(stderr, "Example: %s 10000 128 64 10 32 5 static 16\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]), D = atoi(argv[2]), H = atoi(argv[3]), C = atoi(argv[4]), B = atoi(argv[5]), epochs = atoi(argv[6]);
    const char *sch_type = argv[7];
    int chunk = atoi(argv[8]);

    // TODO 2: Select OpenMP schedule at runtime
    if (strcmp(sch_type, "static") == 0) omp_set_schedule(omp_sched_static, chunk);
    else if (strcmp(sch_type, "dynamic") == 0) omp_set_schedule(omp_sched_dynamic, chunk);
    else if (strcmp(sch_type, "guided") == 0) omp_set_schedule(omp_sched_guided, chunk);

    double learning_rate = 0.05;
    double *X = malloc(N * D * sizeof(double)); 
    int *y = malloc(N * sizeof(int));
    double *W1 = malloc(D * H * sizeof(double)), *W2 = malloc(H * C * sizeof(double));
    double *b1 = calloc(H, sizeof(double)), *b2 = calloc(C, sizeof(double));
    double *dW1 = malloc(D * H * sizeof(double)), *dW2 = malloc(H * C * sizeof(double));
    double *db1 = malloc(H * sizeof(double)), *db2 = malloc(C * sizeof(double));
    double *Z1 = malloc(B * H * sizeof(double)), *A1 = malloc(B * H * sizeof(double));
    double *Z2 = malloc(B * C * sizeof(double)), *P = malloc(B * C * sizeof(double));

    generate_data(X, y, N, D, C);
    initialize_weights(W1, D * H); initialize_weights(W2, H * C);

    double start_time = now_seconds();
    double final_loss = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss_sum = 0.0;
        int epoch_samples = 0;

        for (int batch_start = 0; batch_start < N; batch_start += B) {
            int Bcur = min_int(B, N - batch_start);
            memset(dW1, 0, D*H*sizeof(double)); memset(dW2, 0, H*C*sizeof(double));
            memset(db1, 0, H*sizeof(double)); memset(db2, 0, C*sizeof(double));

            double batch_loss_sum = 0.0;

            // TODO 4: Start Parallel Region per mini-batch
            #pragma omp parallel
            {
                // TODO 5: Allocate private gradient buffers[cite: 2]
                double *ldW1 = calloc(D * H, sizeof(double));
                double *ldW2 = calloc(H * C, sizeof(double));
                double *ldb1 = calloc(H, sizeof(double));
                double *ldb2 = calloc(C, sizeof(double));
                double l_loss = 0.0;

                // TODO 6: Parallelize Sample Loop with runtime scheduling[cite: 2]
                #pragma omp for schedule(runtime)
                for (int i = 0; i < Bcur; i++) {
                    int sample = batch_start + i;
                    // Forward Pass: Hidden
                    for (int h = 0; h < H; h++) {
                        double sum = b1[h];
                        for (int d = 0; d < D; d++) sum += X[IDX2(sample, d, D)] * W1[IDX2(d, h, H)];
                        Z1[IDX2(i, h, H)] = sum;
                        A1[IDX2(i, h, H)] = sum > 0.0 ? sum : 0.0;
                    }
                    // Forward Pass: Output
                    for (int c = 0; c < C; c++) {
                        double sum = b2[c];
                        for (int h = 0; h < H; h++) sum += A1[IDX2(i, h, H)] * W2[IDX2(h, c, C)];
                        Z2[IDX2(i, c, C)] = sum;
                    }
                    // Softmax
                    double max_logit = -DBL_MAX;
                    for (int c = 0; c < C; c++) if (Z2[IDX2(i, c, C)] > max_logit) max_logit = Z2[IDX2(i, c, C)];
                    double denom = 0.0;
                    for (int c = 0; c < C; c++) {
                        double e = exp(Z2[IDX2(i, c, C)] - max_logit);
                        P[IDX2(i, c, C)] = e; denom += e;
                    }
                    for (int c = 0; c < C; c++) P[IDX2(i, c, C)] /= denom;

                    // Loss & Backprop: Output Layer (TODO 8: Private accumulation)[cite: 2]
                    int label = y[sample];
                    l_loss += -log(P[IDX2(i, label, C)] + 1e-12);
                    for (int c = 0; c < C; c++) {
                        double dz2 = (c == label) ? (P[IDX2(i, c, C)] - 1.0) : P[IDX2(i, c, C)];
                        dz2 /= (double)Bcur;
                        ldb2[c] += dz2;
                        for (int h = 0; h < H; h++) ldW2[IDX2(h, c, C)] += A1[IDX2(i, h, H)] * dz2;
                    }
                    // Backprop: Hidden Layer
                    for (int h = 0; h < H; h++) {
                        double da1 = 0.0;
                        for (int c = 0; c < C; c++) {
                            double dz2 = (c == label) ? (P[IDX2(i, c, C)] - 1.0) : P[IDX2(i, c, C)];
                            da1 += (dz2 / Bcur) * W2[IDX2(h, c, C)];
                        }
                        double dz1 = Z1[IDX2(i, h, H)] > 0.0 ? da1 : 0.0;
                        ldb1[h] += dz1;
                        for (int d = 0; d < D; d++) ldW1[IDX2(d, h, H)] += X[IDX2(sample, d, D)] * dz1;
                    }
                }
                // TODO 9: Reduce private results safely[cite: 2]
                #pragma omp critical
                {
                    batch_loss_sum += l_loss;
                    for (int k = 0; k < D*H; k++) dW1[k] += ldW1[k];
                    for (int k = 0; k < H*C; k++) dW2[k] += ldW2[k];
                    for (int k = 0; k < H; k++) db1[k] += ldb1[k];
                    for (int k = 0; k < C; k++) db2[k] += ldb2[k];
                }
                // TODO 10: Free private buffers[cite: 2]
                free(ldW1); free(ldW2); free(ldb1); free(ldb2);
            }

            epoch_loss_sum += batch_loss_sum;
            epoch_samples += Bcur;

            // TODO 11: Update weights after reduction[cite: 2]
            for (int i = 0; i < D * H; i++) W1[i] -= learning_rate * dW1[i];
            for (int i = 0; i < H * C; i++) W2[i] -= learning_rate * dW2[i];
            for (int i = 0; i < H; i++) b1[i] -= learning_rate * db1[i];
            for (int i = 0; i < C; i++) b2[i] -= learning_rate * db2[i];
        }
        final_loss = epoch_loss_sum / (double)epoch_samples;
        printf("Epoch %d/%d - loss = %.8f\n", epoch + 1, epochs, final_loss);
    }

    double elapsed = now_seconds() - start_time;
    double accuracy = compute_accuracy(X, y, W1, W2, b1, b2, N, D, H, C);

    printf("\nParallel result (OpenMP)\n");
    printf("Final loss : %.8f\n", final_loss);
    printf("Accuracy   : %.4f\n", accuracy);
    printf("Time       : %.6f seconds\n", elapsed);

    free(X); free(y); free(W1); free(W2); free(b1); free(b2);
    free(dW1); free(dW2); free(db1); free(db2); free(Z1); free(A1); free(Z2); free(P);
    return 0;
}