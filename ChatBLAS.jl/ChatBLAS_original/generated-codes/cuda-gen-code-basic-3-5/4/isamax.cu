#include "chatblas_cuda.h"

int chatblas_isamax(int n, float *x) {
  // Declare variables
  int index_max = 0;
  float max_val = 0.0;

  // Loop through the vector
  for (int i = 0; i < n; i++) {
    // Calculate the absolute value of the current element
    float abs_val = fabs(x[i]);
    
    // Update the maximum value and index if necessary
    if (abs_val > max_val) {
      max_val = abs_val;
      index_max = i;
    }
  }

  return index_max;
}
