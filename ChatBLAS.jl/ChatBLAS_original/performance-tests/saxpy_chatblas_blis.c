#include <sys/time.h>
#include <stdio.h>
#include "chatblas_openmp.h"
//#include "mkl.h"
//#include "cblas.h"
#include "blis.h"

int main()
{

  struct timeval stop, start;    
  float *chat_x, *chat_y, *chat_x_warm, *chat_y_warm;

  num_t dt;
  dim_t m, n;
  inc_t rs, cs;
  obj_t balpha;
  obj_t x, y;

  dt = BLIS_FLOAT;

  int N = 500000000;
  m = 1; n = 500000000; rs = 0; cs = 0;

  bli_obj_create_1x1( dt, &balpha );
  bli_obj_create( dt, m, n, rs, cs, &x );
  bli_obj_create( dt, m, n, rs, cs, &y );
  
  //bli_printm( "x := x + alpha * w", &x, "%4.1f", "" );


  chat_x = (float *) malloc(N * sizeof(float));
  chat_x_warm = (float *) malloc(N * sizeof(float));
  chat_y = (float *) malloc(N * sizeof(float));
  chat_y_warm = (float *) malloc(N * sizeof(float));
  
  for (int i=0; i<N; ++i) {
    chat_x[i] = 1.0;
    chat_x_warm[i] = 1.0;
    chat_y[i] = 1.0;
    chat_y_warm[i] = 1.0;
  }
  
  bli_setv( &BLIS_ONE, &x );
  bli_setv( &BLIS_ONE, &y );
  
  float alpha = 5.0;
  bli_setsc( 5.0, 0.0, &balpha );
  
  //Warming
  bli_axpyv( &balpha, &y, &x );
 
  gettimeofday(&start, NULL);
  
  bli_axpyv( &balpha, &y, &x );
    
  gettimeofday(&stop, NULL);
  printf("BLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
  
  //Warming
  chatblas_saxpy(N, alpha, chat_x_warm, chat_y_warm);

  gettimeofday(&start, NULL);

  chatblas_saxpy(N, alpha, chat_x, chat_y);

  gettimeofday(&stop, NULL);
  printf("chatBLAS took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  /*
  for (int i=0; i<N; ++i) {
    if(blas_y[i] != chat_y[i]){
      printf("Error\n");
      printf(" BLAS[%d]=%2.f\n ChatBLAS[%d]=%2.f\n",i,blas_y[i],i,chat_y[i]);
      break;
    }
  }
  */
	  
  bli_obj_free( &balpha );
  bli_obj_free( &x );
  bli_obj_free( &y );
  free(chat_y);
  free(chat_y_warm);
  free(chat_x);
  free(chat_x_warm);

  return 0;
}
