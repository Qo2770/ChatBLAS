#include <sys/time.h>
#include <cublas_v2.h>
#include "chatblas_cuda.h"

int main()
{

  struct timeval stop, start;    
  float *host_x, *host_y_cublas, *host_y_chatblas;
  float *dev_x, *dev_y;
  cublasHandle_t h;

  int N = 500000000;
  host_x =          (float *) malloc(N * sizeof(float));
  host_y_cublas = (float *) malloc(N * sizeof(float));
  host_y_chatblas = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    host_x[i] = 2.0;
    host_y_cublas[i] = 3.0;
    host_y_chatblas[i] = 3.0;
  }
  
  float alpha = 5.0;
 
  cublasCreate(&h);
  
  gettimeofday(&start, NULL);
  
  cudaMalloc( (void**)&dev_x, N*sizeof(float));
  cudaMalloc( (void**)&dev_y, N*sizeof(float));

  cublasSetVector(N, sizeof(host_x[0]), host_x, 1, dev_x, 1);
  cublasSetVector(N, sizeof(host_y_cublas[0]), host_y_cublas, 1, dev_y, 1);
  cudaDeviceSynchronize();

  cublasSaxpy(h, N, &alpha, dev_x, 1, dev_y, 1);
  cudaDeviceSynchronize();

  cublasGetVector(N, sizeof(host_y_cublas[0]), dev_y, 1, host_y_cublas, 1);
  cudaDeviceSynchronize();

  gettimeofday(&stop, NULL);
  printf("cuBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  if (h)
    cublasDestroy(h);
  cudaFree(dev_y);
  cudaFree(dev_x);

  gettimeofday(&start, NULL);

  chatblas_saxpy(N, alpha, host_x, host_y_chatblas);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  for (int i=0; i<N; ++i) {
    if(host_y_chatblas[i] != host_y_cublas[i]){
      printf("Error\n");
      break;
    }
  }
	    
	  
  free(host_y_cublas);
  free(host_y_chatblas);
  free(host_x);

  return 0;
}
