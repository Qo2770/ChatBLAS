#include <stdio.h>
#include <math.h>
#include <chatblas_openmp.h>

int chatblas_isamax(int n, float *x) {
    int maxPos = 0;
    float maxValue = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float absValue = fabs(x[i]);
        #pragma omp critical
        {
            if (absValue > maxValue) {
                maxValue = absValue;
                maxPos = i;
            }
        }
    }

    return maxPos;
}
