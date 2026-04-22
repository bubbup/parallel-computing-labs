#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static long local_sum_range(long start, long end) {
long s = 0;
for (long i = start; i < end; i++) {
s += i;
}
return s;
}

int main(int argc, char **argv) {
int rank, size;
/* Task A: MPI Hello World */
/* TODO A1: initialize MPI */
MPI_Init(&argc, &argv);
/* TODO A2: get rank */
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
/* TODO A3: get size */
MPI_Comm_size(MPI_COMM_WORLD, &size); 
printf("Hello from rank %d of %d\n", rank, size);
/* TODO A4: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
/* Task B: Point-to-Point */
if (rank == 0) {
printf("\n=== Task B: Point-to-Point ===\n");
}
/* TODO B1: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
int value = 42;
if (size >= 2) {
if (rank == 0) {
/* TODO B2: send value from rank 0 to rank 1 (Solved) */
MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
printf("Rank 0 sent %d to rank 1\n", value);
} else if (rank == 1) {
int received = -1;
/* TODO B3: receive value at rank 1 from rank 0 */
MPI_Recv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
printf("Rank 1 received %d\n", received);
}
}
else {
if (rank == 0) {
printf("Need at least 2 processes for point-to-point test\n");
}
}
/* TODO B4: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
/* Task C: MPI Ring Communication */
if (rank == 0) {
printf("\n=== Task C: Ring Communication ===\n");
}
/* TODO C1: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
if (size >= 2) {
int token = -1;
int next = (rank + 1) % size;
int prev = (rank - 1 + size) % size;
if (rank == 0) {
token = 100;
/* TODO C2: send token to next process */
MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
/* TODO C3: receive token back from previous process */
MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
printf("Rank 0 got token back = %d\n", token);
}
else {
/* TODO C4: receive token from previous process */
MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
token += rank;
/* TODO C5: send token to next process */
MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
printf("Rank %d forwarded token = %d\n", rank, token);
}
} else {
if (rank == 0) {
printf("Need at least 2 processes for ring test\n");
}
}
/* TODO C6: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
/* Task D1: MPI Broadcast */
MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);
if (rank == 0) {
printf("\n=== Task D1: Broadcast ===\n");
}
/* TODO D1: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
long N = 0;
if (rank == 0) {
N = (argc >= 2) ? atol(argv[1]) : 1000000;
}
/* TODO D2: broadcast N from rank 0 to all ranks */
MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);
printf("Rank %d sees N = %ld\n", rank, N);
/* TODO D3: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);



/* Task D2: MPI Reduce */
if (rank == 0) {
printf("\n=== Task D2: Reduce ===\n");
}
/* TODO D4: synchronize all processes */
MPI_Barrier(MPI_COMM_WORLD);
long chunk = N / size;
long start = (long)rank * chunk;
long end = (rank == size - 1) ? N : start + chunk;
long local = local_sum_range(start, end);
long global = 0;
/* TODO D5: reduce all local sums into global sum at rank 0 */
MPI_Reduce(&local, &global, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD); [cite: 100, 144]
printf("Rank %d: local sum over [%ld, %ld) = %ld\n", rank, start, end, local);
if (rank == 0) {
long expected = (N * (N - 1)) / 2;
printf("Global sum = %ld\n", global);
printf("Expected sum = %ld\n", expected);
printf("Correct? = %s\n", (global == expected) ? "yes" : "no");
}
/* TODO D6: finalize MPI */
MPI_Finalize();
return 0;
}
