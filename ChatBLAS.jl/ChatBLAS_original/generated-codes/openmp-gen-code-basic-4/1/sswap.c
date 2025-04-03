#include "chatblas_openmp.h"
#include <omp.h>

void chatblas_sswap(int n, float *x, float *y)
{
    #pragma omp parallel
    {
        int i;
        float temp;
        #pragma omp for schedule(static) 
        for(i=0; i<n; i++)
        {
            temp = x[i];
            x[i] = y[i];
            y[i] = temp;
        }
    }
}
