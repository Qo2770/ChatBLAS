#include <sys/time.h>
#include <hipblas.h>
#include "chatblas_hip.h"

int main()
{

  struct timeval stop, start;    
  float *host_x;
  float *dev_x, *dev_result;
  float result_hipblas, result_chatblas;
  hipblasHandle_t h;

  int N = 500000000;
  host_x = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    host_x[i] = 1.0;
  }
  
  hipblasCreate(&h);
  
  gettimeofday(&start, NULL);
  
  hipMalloc( (void**)&dev_x, N*sizeof(float));
  hipMalloc( (void**)&dev_result, sizeof(float));

  hipblasSetVector(N, sizeof(host_x[0]), host_x, 1, dev_x, 1);
  hipDeviceSynchronize();

  hipblasSasum(h, N, dev_x, 1, dev_result);
  hipDeviceSynchronize();

  hipMemcpy(&result_hipblas, dev_result, sizeof(float), hipMemcpyDeviceToHost);

  gettimeofday(&stop, NULL);
  printf("hipBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  if (h)
    hipblasDestroy(h);
  hipFree(dev_x);

  gettimeofday(&start, NULL);

  result_chatblas = chatblas_sasum( N, host_x );

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  if(result_chatblas != result_hipblas){
    printf("Error, hipBLAS = %2.f, chatBLAS = %2.f\n", result_hipblas, result_chatblas);
  }
	  
  free(host_x);

  return 0;
}
