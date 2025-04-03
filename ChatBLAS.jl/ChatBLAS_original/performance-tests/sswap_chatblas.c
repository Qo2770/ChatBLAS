#include <sys/time.h>
#include <stdio.h>
#include "chatblas_openmp.h"
//#include "mkl.h"
#include "cblas.h"

int main()
{

  struct timeval stop, start;    
  float *blas_x, *blas_y, *blas_x_warm, *blas_y_warm;
  float *chat_x, *chat_y, *chat_x_warm, *chat_y_warm;

  int N = 500000000;
  blas_x = (float *) malloc(N * sizeof(float));
  blas_x_warm = (float *) malloc(N * sizeof(float));
  chat_x = (float *) malloc(N * sizeof(float));
  chat_x_warm = (float *) malloc(N * sizeof(float));
  blas_y = (float *) malloc(N * sizeof(float));
  blas_y_warm = (float *) malloc(N * sizeof(float));
  chat_y = (float *) malloc(N * sizeof(float));
  chat_y_warm = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    blas_x[i] = 2.0;
    blas_x_warm[i] = 2.0;
    chat_x[i] = 2.0;
    chat_x_warm[i] = 2.0;
    blas_y[i] = 3.0;
    blas_y_warm[i] = 3.0;
    chat_y[i] = 3.0;
    chat_y_warm[i] = 3.0;
  }
  
  float alpha = 5.0;
  
  //Warming
  cblas_sswap(N, blas_x_warm, 1, blas_y_warm, 1);
 
  gettimeofday(&start, NULL);
  
  cblas_sswap(N, blas_x, 1, blas_y, 1);
    
  gettimeofday(&stop, NULL);
  printf("BLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
  
  //Warming
  chatblas_sswap(N, chat_x_warm, chat_y_warm);

  gettimeofday(&start, NULL);

  chatblas_sswap(N, chat_x, chat_y);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  for (int i=0; i<N; ++i) {
    if(blas_y[i] != chat_y[i]){
      printf("Error\n");
      printf(" BLAS[%d]=%2.f\n ChatBLAS[%d]=%2.f\n",i,blas_y[i],i,chat_y[i]);
      break;
    }
  }
	  
  free(blas_y);
  free(blas_y_warm);
  free(chat_y);
  free(chat_y_warm);
  free(blas_x);
  free(blas_x_warm);
  free(chat_x);
  free(chat_x_warm);

  return 0;
}
