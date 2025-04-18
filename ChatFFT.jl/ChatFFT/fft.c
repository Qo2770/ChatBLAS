
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "chatfft_openmp.h"

// Function prototype
void chatfft(float *x, int n, int isign);

int reverse_bits(int index, int n) {
    int result = 0;
    while (n > 1) {
        result = (result << 1) | (index & 1);
        index >>= 1;
        n >>= 1;
    }
    return result;
}

void chatfft(float *x, int n, int isign) {
    int i, j, k, m;
    int step;
    float wr, wi, wpr, wpi, theta, temp_r, temp_i;
    
    // Bit-reversal reordering
    #pragma omp parallel for private(j, m)
    for (i = 0; i < n; ++i) {
        j = reverse_bits(i, n);
        if (j > i) {
            temp_r = x[2 * i];
            temp_i = x[2 * i + 1];
            x[2 * i] = x[2 * j];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j] = temp_r;
            x[2 * j + 1] = temp_i;
        }
    }
    
    // FFT implementation: Cooley-Tukey radix-2
    for (step = 2; step <= n; step <<= 1) {
        theta = isign * 2.0f * M_PI / step;
        wpr = cos(theta);
        wpi = sin(theta);
        
        #pragma omp parallel for private(m, wr, wi, temp_r, temp_i, j, k)
        for (m = 0; m < n; m += step) {
            wr = 1.0f;
            wi = 0.0f;
            for (k = 0; k < step / 2; ++k) {
                j = m + k;
                temp_r = wr * x[2 * (j + step / 2)] - wi * x[2 * (j + step / 2) + 1];
                temp_i = wr * x[2 * (j + step / 2) + 1] + wi * x[2 * (j + step / 2)];
                x[2 * (j + step / 2)] = x[2 * j] - temp_r;
                x[2 * (j + step / 2) + 1] = x[2 * j + 1] - temp_i;
                x[2 * j] += temp_r;
                x[2 * j + 1] += temp_i;
                
                // Update the twiddle factors
                temp_r = wr;
                wr = temp_r * wpr - wi * wpi;
                wi = temp_r * wpi + wi * wpr;
            }
        }
    }
    
    // Normalize if inverse transform
    if (isign == 1) {
        #pragma omp parallel for
        for (i = 0; i < 2 * n; ++i) {
            x[i] /= n;
        }
    }
}
