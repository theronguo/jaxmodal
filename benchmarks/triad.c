/* Multithreaded STREAM triad a = b + s*c. Usage: triad <threads>. Prints "<GB/s> <checksum>". */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 20000000L
#define REPS 20

static double *a, *b, *c;
static int nthreads, waiting;
static long generation;
static double stamp[REPS + 2];
static int nstamp;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* The last thread to arrive records the time, so stamps don't depend on wake-up order. */
static void barrier(void) {
    pthread_mutex_lock(&mu);
    long g = generation;
    if (++waiting == nthreads) {
        stamp[nstamp++] = now();
        waiting = 0;
        generation++;
        pthread_cond_broadcast(&cv);
    } else {
        while (g == generation) pthread_cond_wait(&cv, &mu);
    }
    pthread_mutex_unlock(&mu);
}

static void *work(void *arg) {
    long t = (long)arg, lo = N * t / nthreads, hi = N * (t + 1) / nthreads;
    for (long i = lo; i < hi; i++) a[i] = 0.0, b[i] = 1.0, c[i] = 2.0;  /* first touch */
    barrier();
    for (int r = 0; r < REPS; r++) {
        for (long i = lo; i < hi; i++) a[i] = b[i] + 3.0 * c[i];
        barrier();
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    nthreads = atoi(argv[1]);
    a = malloc(N * sizeof(double));
    b = malloc(N * sizeof(double));
    c = malloc(N * sizeof(double));
    pthread_t th[nthreads];
    for (long t = 0; t < nthreads; t++) pthread_create(&th[t], NULL, work, (void *)t);
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
    double best = 1e30, sum = 0.0;
    for (int r = 0; r < REPS; r++) {
        double dt = stamp[r + 1] - stamp[r];
        if (dt < best) best = dt;
    }
    for (long i = 0; i < N; i++) sum += a[i];
    printf("%.3f %.17g\n", 24.0 * N / best / 1e9, sum);
    return 0;
}
