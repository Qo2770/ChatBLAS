#include <sys/time.h>
#include <stdio.h>
#include <fftw3.h>
#include "chatfft_openmp.h"
//#include "mkl.h"
#include <stdlib.h>

#define NUM_POINTS 65536
#define REAL 0
#define IMAG 1

void build_input(fftw_complex* signal) {

    int i;
    for (i = 0; i < NUM_POINTS; ++i) {
        double theta = (double)i / (double)NUM_POINTS * M_PI;

        signal[i][REAL] = 1.0 * cos(10.0 * theta) +
                          0.5 * cos(25.0 * theta);

        signal[i][IMAG] = 1.0 * sin(10.0 * theta) +
                          0.5 * sin(25.0 * theta);
    }
}

void build_input_fl(float* signal) {

    int i;
    for (i = 0; i < NUM_POINTS; ++i) {
        float theta = (float)i / (float)NUM_POINTS * M_PI;

        signal[i*2] = 1.0 * cos(10.0 * theta) +
                          0.5 * cos(25.0 * theta);

        signal[i*2+1] = 1.0 * sin(10.0 * theta) +
                          0.5 * sin(25.0 * theta);
    }
}

int main()
{

  struct timeval stop, start;    
  float *chat_x, *chat_x_warm;

  chat_x = (float *) malloc(2 * NUM_POINTS * sizeof(float));
  chat_x_warm = (float *) malloc(2 * NUM_POINTS * sizeof(float));
  
  fftw_complex signal_warm[NUM_POINTS];
  fftw_complex result_warm[NUM_POINTS];
  fftw_complex signal[NUM_POINTS];
  fftw_complex result[NUM_POINTS];
  

  build_input_fl(chat_x);
  build_input_fl(chat_x_warm);

  //Warming
  fftw_plan plan = fftw_plan_dft_1d(NUM_POINTS,
                                      signal_warm,
                                      result_warm,
                                      FFTW_FORWARD,
                                      FFTW_ESTIMATE);

  build_input(signal_warm);
  fftw_execute(plan);

  plan = fftw_plan_dft_1d(NUM_POINTS,
                                      signal,
                                      result,
                                      FFTW_FORWARD,
                                      FFTW_ESTIMATE);
 
  build_input(signal);
  gettimeofday(&start, NULL);
  
  fftw_execute(plan);

  gettimeofday(&stop, NULL);
  printf("FFTW took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
  
  //Warming
  chatfft(chat_x_warm, NUM_POINTS, -1);

  gettimeofday(&start, NULL);

  chatfft(chat_x, NUM_POINTS, -1);

  gettimeofday(&stop, NULL);
  printf("chatFFT took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

  free(chat_x);
  free(chat_x_warm);

  return 0;
}
