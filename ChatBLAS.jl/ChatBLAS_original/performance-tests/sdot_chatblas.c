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
  float blas_result = 0.0; 
  float chat_result = 0.0;

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
    blas_x[i] = 1.0;
    blas_x_warm[i] = 1.0;
    chat_x[i] = 1.0;
    chat_x_warm[i] = 1.0;
    blas_y[i] = 0.01;
    blas_y_warm[i] = 0.01;
    chat_y[i] = 0.01;
    chat_y_warm[i] = 0.01;
  }
 
  //Warming 
  blas_result = cblas_sdot(N, blas_x_warm, 1, blas_y_warm, 1);
  
  gettimeofday(&start, NULL);
  
  blas_result = cblas_sdot(N, blas_x, 1, blas_y, 1);
    
  gettimeofday(&stop, NULL);
  printf("BLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
  
  //Warming 
  chat_result = chatblas_sdot(N, chat_x_warm, chat_y_warm);

  gettimeofday(&start, NULL);

  chat_result = chatblas_sdot(N, chat_x, chat_y);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  for (int i=0; i<N; ++i) {
    if(blas_result != chat_result){
      printf("Error\n");
      printf(" BLAS=%2.f\n ChatBLAS=%2.f\n", blas_result, chat_result);
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
