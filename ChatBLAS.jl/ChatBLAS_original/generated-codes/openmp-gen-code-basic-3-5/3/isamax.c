#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int pos = 0;
    float max_val = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float abs_val = (x[i] < 0) ? -x[i] : x[i];
        #pragma omp critical
        {
            if (abs_val > max_val) {
                max_val = abs_val;
                pos = i;
            }
        }
    }

    return pos;
}
