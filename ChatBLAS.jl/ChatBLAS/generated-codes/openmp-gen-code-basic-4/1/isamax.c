#include "chatblas_openmp.h"
#include <math.h>

int chatblas_isamax(int n, float *x) {
    int max_pos = -1;
    float max_val = 0.0;
    #pragma omp parallel
    {
        float local_max = 0.0;
        int local_pos = -1;
    
        #pragma omp for
        for(int i = 0; i < n; i++) {
            if(fabs(x[i]) > local_max) {
                local_max = fabs(x[i]);
                local_pos = i;
            }
        }
        
        // Critical section to update the maximum value with thread-safety
        #pragma omp critical
        {
            if(local_max > max_val) {
                max_val = local_max;
                max_pos = local_pos;
            }
        }
    }
    return max_pos;
}
