#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int maxPos = 0;
    float maxValue = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp critical
        {
            if (fabs(x[i]) > fabs(maxValue)) {
                maxPos = i;
                maxValue = x[i];
            }
        }
    }

    return maxPos;
}
