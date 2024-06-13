#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int i, pos, max_pos;
    float max_val;

    max_pos = 0;
    max_val = 0.0;

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        if (fabs(x[i]) > max_val) {
            #pragma omp critical
            {
                if (fabs(x[i]) > max_val) {
                    max_val = fabs(x[i]);
                    max_pos = i;
                }
            }
        }
    }

    return max_pos;
}