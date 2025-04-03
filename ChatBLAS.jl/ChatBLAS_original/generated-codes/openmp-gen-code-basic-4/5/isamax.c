#include <math.h>
#include <omp.h>
#include "chatblas_openmp.h"

int chatblas_isamax(int n, float *x) 
{
    int max_idx = -1;
    float max_val = -1.0f;

    #pragma omp parallel for 
    for(int i = 0; i < n; i++) 
    {
        float abs_val = fabs(x[i]);

        #pragma omp critical 
        {
            if(abs_val > max_val) 
            {
                max_val = abs_val;
                max_idx = i;
            }
        }
    }

    return max_idx;
}
