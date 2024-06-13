#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) {
    int i, max_index;
    float max_value = -1.0;  // Initialize with negative value
    
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        float abs_value = fabsf(x[i]);
        // Compare and update max value and index
        #pragma omp critical
        {
            if (abs_value > max_value) {
                max_value = abs_value;
                max_index = i;
            }
        }
    }
    
    return max_index;
}
