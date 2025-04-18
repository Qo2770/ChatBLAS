#include <math.h>
#include <omp.h>

#define M_PI 3.14159265359

//BLAS Level 1
void chatfft(float *x, int n, int isign);
