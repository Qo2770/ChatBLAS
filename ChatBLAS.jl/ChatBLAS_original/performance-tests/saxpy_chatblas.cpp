#include <sys/time.h>
#include <hipblas.h>
#include "chatblas_hip.h"

int main()
{

  struct timeval stop, start;    
  float *host_x, *host_y_hipblas, *host_y_chatblas;
  float *dev_x, *dev_y;
  hipblasHandle_t h;

  int N = 500000000;
  host_x =          (float *) malloc(N * sizeof(float));
  host_y_hipblas = (float *) malloc(N * sizeof(float));
  host_y_chatblas = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    host_x[i] = 2.0;
    host_y_hipblas[i] = 3.0;
    host_y_chatblas[i] = 3.0;
  }
  
  float alpha = 5.0;
 
  hipblasCreate(&h);
  
  gettimeofday(&start, NULL);
  
  hipMalloc( (void**)&dev_x, N*sizeof(float));
  hipMalloc( (void**)&dev_y, N*sizeof(float));

  hipblasSetVector(N, sizeof(host_x[0]), host_x, 1, dev_x, 1);
  hipblasSetVector(N, sizeof(host_y_hipblas[0]), host_y_hipblas, 1, dev_y, 1);
  hipDeviceSynchronize();

  hipblasSaxpy(h, N, &alpha, dev_x, 1, dev_y, 1);
  hipDeviceSynchronize();

  hipblasGetVector(N, sizeof(host_y_hipblas[0]), dev_y, 1, host_y_hipblas, 1);
  hipDeviceSynchronize();

  gettimeofday(&stop, NULL);
  printf("hipBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  if (h)
    hipblasDestroy(h);
  hipFree(dev_y);
  hipFree(dev_x);

  gettimeofday(&start, NULL);

  chatblas_saxpy(N, alpha, host_x, host_y_chatblas);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  for (int i=0; i<N; ++i) {
    if(host_y_chatblas[i] != host_y_hipblas[i]){
      printf("Error\n");
      break;
    }
  }
	    
	  
  free(host_y_hipblas);
  free(host_y_chatblas);
  free(host_x);

  return 0;
}
