/*
  nn_seq.c
  Sequential baseline for a small two-layer neural network.

  Network:
      X -> W1 -> ReLU -> W2 -> Softmax

  Compile:
      gcc -O2 nn_seq.c -o nn_seq -lm

  Run:
      ./nn_seq N D H C B epochs

  Example:
      ./nn_seq 10000 128 64 10 32 5
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define IDX2(i, j, cols) ((i) * (cols) + (j))

static double now_seconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
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
            X[IDX2(i, j, D)] =
                sin(0.01 * i + 0.02 * j) + 0.1 * (double)y[i];
        }
    }
}

static void initialize_weights(double *W, int size) {
    for (int i = 0; i < size; i++) {
        W[i] = rand_small(i);
    }
}

static void zero_array(double *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = 0.0;
    }
}

static int predict_one(
    const double *X,
    const double *W1,
    const double *W2,
    const double *b1,
    const double *b2,
    int sample,
    int D,
    int H,
    int C
) {
    double *A1 = (double *)malloc((size_t)H * sizeof(double));
    double *Z2 = (double *)malloc((size_t)C * sizeof(double));

    if (!A1 || !Z2) {
        fprintf(stderr, "Allocation failed in predict_one.\n");
        exit(1);
    }

    for (int h = 0; h < H; h++) {
        double sum = b1[h];

        for (int d = 0; d < D; d++) {
            sum += X[IDX2(sample, d, D)] * W1[IDX2(d, h, H)];
        }

        A1[h] = sum > 0.0 ? sum : 0.0;
    }

    int best_class = 0;
    double best_score = -DBL_MAX;

    for (int c = 0; c < C; c++) {
        double sum = b2[c];

        for (int h = 0; h < H; h++) {
            sum += A1[h] * W2[IDX2(h, c, C)];
        }

        Z2[c] = sum;

        if (sum > best_score) {
            best_score = sum;
            best_class = c;
        }
    }

    free(A1);
    free(Z2);

    return best_class;
}

static double compute_accuracy(
    const double *X,
    const int *y,
    const double *W1,
    const double *W2,
    const double *b1,
    const double *b2,
    int N,
    int D,
    int H,
    int C
) {
    int correct = 0;

    for (int i = 0; i < N; i++) {
        int pred = predict_one(X, W1, W2, b1, b2, i, D, H, C);

        if (pred == y[i]) {
            correct++;
        }
    }

    return (double)correct / (double)N;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s N D H C B epochs\n", argv[0]);
        fprintf(stderr, "Example: %s 10000 128 64 10 32 5\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int D = atoi(argv[2]);
    int H = atoi(argv[3]);
    int C = atoi(argv[4]);
    int B = atoi(argv[5]);
    int epochs = atoi(argv[6]);

    if (N <= 0 || D <= 0 || H <= 0 || C <= 1 || B <= 0 || epochs <= 0) {
        fprintf(stderr, "All dimensions must be positive, and C must be greater than 1.\n");
        return 1;
    }

    double learning_rate = 0.05;

    double *X = (double *)malloc((size_t)N * D * sizeof(double));
    int *y = (int *)malloc((size_t)N * sizeof(int));

    double *W1 = (double *)malloc((size_t)D * H * sizeof(double));
    double *W2 = (double *)malloc((size_t)H * C * sizeof(double));
    double *b1 = (double *)calloc((size_t)H, sizeof(double));
    double *b2 = (double *)calloc((size_t)C, sizeof(double));

    double *dW1 = (double *)calloc((size_t)D * H, sizeof(double));
    double *dW2 = (double *)calloc((size_t)H * C, sizeof(double));
    double *db1 = (double *)calloc((size_t)H, sizeof(double));
    double *db2 = (double *)calloc((size_t)C, sizeof(double));

    double *Z1 = (double *)malloc((size_t)B * H * sizeof(double));
    double *A1 = (double *)malloc((size_t)B * H * sizeof(double));
    double *Z2 = (double *)malloc((size_t)B * C * sizeof(double));
    double *P = (double *)malloc((size_t)B * C * sizeof(double));

    if (!X || !y || !W1 || !W2 || !b1 || !b2 ||
        !dW1 || !dW2 || !db1 || !db2 || !Z1 || !A1 || !Z2 || !P) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    generate_data(X, y, N, D, C);
    initialize_weights(W1, D * H);
    initialize_weights(W2, H * C);

    double start_time = now_seconds();
    double final_loss = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss_sum = 0.0;
        int epoch_samples = 0;

        for (int batch_start = 0; batch_start < N; batch_start += B) {
            int Bcur = min_int(B, N - batch_start);

            zero_array(dW1, D * H);
            zero_array(dW2, H * C);
            zero_array(db1, H);
            zero_array(db2, C);

            /*
              Forward pass: hidden layer.
            */
            for (int i = 0; i < Bcur; i++) {
                int sample = batch_start + i;

                for (int h = 0; h < H; h++) {
                    double sum = b1[h];

                    for (int d = 0; d < D; d++) {
                        sum += X[IDX2(sample, d, D)] * W1[IDX2(d, h, H)];
                    }

                    Z1[IDX2(i, h, H)] = sum;
                    A1[IDX2(i, h, H)] = sum > 0.0 ? sum : 0.0;
                }
            }

            /*
              Forward pass: output logits.
            */
            for (int i = 0; i < Bcur; i++) {
                for (int c = 0; c < C; c++) {
                    double sum = b2[c];

                    for (int h = 0; h < H; h++) {
                        sum += A1[IDX2(i, h, H)] * W2[IDX2(h, c, C)];
                    }

                    Z2[IDX2(i, c, C)] = sum;
                }
            }

            /*
              Softmax.
            */
            for (int i = 0; i < Bcur; i++) {
                double max_logit = -DBL_MAX;

                for (int c = 0; c < C; c++) {
                    double z = Z2[IDX2(i, c, C)];

                    if (z > max_logit) {
                        max_logit = z;
                    }
                }

                double denom = 0.0;

                for (int c = 0; c < C; c++) {
                    double e = exp(Z2[IDX2(i, c, C)] - max_logit);
                    P[IDX2(i, c, C)] = e;
                    denom += e;
                }

                for (int c = 0; c < C; c++) {
                    P[IDX2(i, c, C)] /= denom;
                }
            }

            /*
              Loss.
            */
            double batch_loss = 0.0;

            for (int i = 0; i < Bcur; i++) {
                int label = y[batch_start + i];
                batch_loss += -log(P[IDX2(i, label, C)] + 1e-12);
            }

            batch_loss /= (double)Bcur;
            epoch_loss_sum += batch_loss * (double)Bcur;
            epoch_samples += Bcur;

            /*
              Backpropagation: output layer.
            */
            for (int i = 0; i < Bcur; i++) {
                int label = y[batch_start + i];

                for (int c = 0; c < C; c++) {
                    double dz2 = P[IDX2(i, c, C)];

                    if (c == label) {
                        dz2 -= 1.0;
                    }

                    dz2 /= (double)Bcur;

                    db2[c] += dz2;

                    for (int h = 0; h < H; h++) {
                        dW2[IDX2(h, c, C)] += A1[IDX2(i, h, H)] * dz2;
                    }
                }
            }

            /*
              Backpropagation: hidden layer.
            */
            for (int i = 0; i < Bcur; i++) {
                int label = y[batch_start + i];

                for (int h = 0; h < H; h++) {
                    double da1 = 0.0;

                    for (int c = 0; c < C; c++) {
                        double dz2 = P[IDX2(i, c, C)];

                        if (c == label) {
                            dz2 -= 1.0;
                        }

                        dz2 /= (double)Bcur;

                        da1 += dz2 * W2[IDX2(h, c, C)];
                    }

                    double dz1 = Z1[IDX2(i, h, H)] > 0.0 ? da1 : 0.0;

                    db1[h] += dz1;

                    for (int d = 0; d < D; d++) {
                        dW1[IDX2(d, h, H)] += X[IDX2(batch_start + i, d, D)] * dz1;
                    }
                }
            }

            /*
              SGD update.
            */
            for (int i = 0; i < D * H; i++) {
                W1[i] -= learning_rate * dW1[i];
            }

            for (int i = 0; i < H * C; i++) {
                W2[i] -= learning_rate * dW2[i];
            }

            for (int i = 0; i < H; i++) {
                b1[i] -= learning_rate * db1[i];
            }

            for (int i = 0; i < C; i++) {
                b2[i] -= learning_rate * db2[i];
            }
        }

        final_loss = epoch_loss_sum / (double)epoch_samples;

        printf("Epoch %d/%d - loss = %.8f\n", epoch + 1, epochs, final_loss);
    }

    double end_time = now_seconds();
    double elapsed = end_time - start_time;

    double accuracy = compute_accuracy(X, y, W1, W2, b1, b2, N, D, H, C);

    printf("\nSequential result\n");
    printf("Final loss : %.8f\n", final_loss);
    printf("Accuracy   : %.4f\n", accuracy);
    printf("Time       : %.6f seconds\n", elapsed);

    free(X);
    free(y);
    free(W1);
    free(W2);
    free(b1);
    free(b2);
    free(dW1);
    free(dW2);
    free(db1);
    free(db2);
    free(Z1);
    free(A1);
    free(Z2);
    free(P);

    return 0;
}