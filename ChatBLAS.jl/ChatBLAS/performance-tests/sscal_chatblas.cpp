#include <sys/time.h>
#include <hipblas.h>
#include "chatblas_hip.h"

int main()
{

  struct timeval stop, start;    
  float *host_x_hipblas, *host_x_chatblas;
  float *dev_x;
  hipblasHandle_t h;

  int N = 500000000;
  host_x_hipblas = (float *) malloc(N * sizeof(float));
  host_x_chatblas = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    host_x_hipblas[i] = 3.0;
    host_x_chatblas[i] = 3.0;
  }
  
  float alpha = 5.0;
 
  hipblasCreate(&h);
  
  gettimeofday(&start, NULL);
  
  hipMalloc( (void**)&dev_x, N*sizeof(float));

  hipblasSetVector(N, sizeof(host_x_hipblas[0]), host_x_hipblas, 1, dev_x, 1);
  hipDeviceSynchronize();

  hipblasSscal(h, N, &alpha, dev_x, 1);
  hipDeviceSynchronize();

  hipblasGetVector(N, sizeof(host_x_hipblas[0]), dev_x, 1, host_x_hipblas, 1);
  hipDeviceSynchronize();

  gettimeofday(&stop, NULL);
  printf("hipBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  if (h)
    hipblasDestroy(h);
  hipFree(dev_x);

  gettimeofday(&start, NULL);

  chatblas_sscal(N, alpha, host_x_chatblas);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  for (int i=0; i<N; ++i) {
    if(host_x_chatblas[i] != host_x_hipblas[i]){
      printf("Error\n");
      break;
    }
  }
	    
	  
  free(host_x_hipblas);
  free(host_x_chatblas);

  return 0;
}
