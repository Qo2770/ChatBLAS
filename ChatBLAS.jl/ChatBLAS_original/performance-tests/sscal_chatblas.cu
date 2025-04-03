#include <sys/time.h>
#include <cublas_v2.h>
#include "chatblas_cuda.h"

int main()
{

  struct timeval stop, start;    
  float *host_x_cublas, *host_x_chatblas;
  float *dev_x;
  cublasHandle_t h;

  int N = 500000000;
  host_x_cublas = (float *) malloc(N * sizeof(float));
  host_x_chatblas = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    host_x_cublas[i] = 3.0;
    host_x_chatblas[i] = 3.0;
  }
  
  float alpha = 5.0;
 
  cublasCreate(&h);
  
  gettimeofday(&start, NULL);
  
  cudaMalloc( (void**)&dev_x, N*sizeof(float));

  cublasSetVector(N, sizeof(host_x_cublas[0]), host_x_cublas, 1, dev_x, 1);
  cudaDeviceSynchronize();

  cublasSscal(h, N, &alpha, dev_x, 1);
  cudaDeviceSynchronize();

  cublasGetVector(N, sizeof(host_x_cublas[0]), dev_x, 1, host_x_cublas, 1);
  cudaDeviceSynchronize();

  gettimeofday(&stop, NULL);
  printf("cuBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  if (h)
    cublasDestroy(h);
  cudaFree(dev_x);

  gettimeofday(&start, NULL);

  chatblas_sscal(N, alpha, host_x_chatblas);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  for (int i=0; i<N; ++i) {
    if(host_x_chatblas[i] != host_x_cublas[i]){
      printf("Error\n");
      break;
    }
  }
	    
	  
  free(host_x_cublas);
  free(host_x_chatblas);

  return 0;
}



