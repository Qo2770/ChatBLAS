#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

//BLAS Level 1
void chatfft(float *x, int n, int isign);

#ifdef __cplusplus
}
#endif
