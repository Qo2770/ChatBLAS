#include "chatblas_openmp.h"
#include <stdlib.h>
#include <omp.h>

void chatblas_scopy(int n, float *x, float *y) {
    // Check if x and y are not null
    if (x == NULL || y == NULL) {
        return;
    }

    // Specify number of threads based on number of cores in your system
    int num_threads = omp_get_max_threads();

    // Perform the copy operation in parallel
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}
