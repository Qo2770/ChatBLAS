#include <stdlib.h>
#include <stdio.h>
//#include "chatblas_hip.h"
#include "chatblas_cuda.h"

int main()
{
  int SIZE = 100;
  float a = 5.0;
  float* x, * y;

  x = (float*)malloc(SIZE * sizeof(float));
  y = (float*)malloc(SIZE * sizeof(float));
  for (int i = 0; i < SIZE; i++)
  {
    x[i] = 5.0;
    y[i] = 1.0;
  }
  
  chatblas_sscal(SIZE, a, x);
  chatblas_sswap(SIZE, x, y);
  
  for (int i = 0; i < SIZE; i++)
    printf("x[%d] = %2.f\n", i, x[i]);

}
