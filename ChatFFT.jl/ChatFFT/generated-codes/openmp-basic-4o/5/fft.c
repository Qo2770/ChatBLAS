
#include "chatfft_openmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void chatfft(float *x, int n, int isign) {
    int i, j, m, mmax, istep;
    float wtemp, wr, wpr, wpi, wi, theta, tempr, tempi;

    // Bit-reversal permutation
    int nv2 = n / 2;
    int nm1 = n - 1;
    j = 0;
    for (i = 0; i < nm1; i++) {
        if (i < j) {
            // Swap real part
            float temp = x[2*i];
            x[2*i] = x[2*j];
            x[2*j] = temp;
            // Swap imaginary part
            temp = x[2*i+1];
            x[2*i+1] = x[2*j+1];
            x[2*j+1] = temp;
        }
        int m = nv2;
        while (m <= j) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // FFT computation using Cooley-Tukey
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);

    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = isign * (2.0 * M_PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        #pragma omp parallel for private(i, j, wr, wi, wtemp, tempr, tempi) schedule(static)
        for (m = 0; m < mmax; m += 2) {
            wr = 1.0;
            wi = 0.0;
            for (i = m; i < n; i += istep) {
                j = i + mmax;
                tempr = wr * x[2*j] - wi * x[2*j+1];
                tempi = wr * x[2*j+1] + wi * x[2*j];
                x[2*j] = x[2*i] - tempr;
                x[2*j+1] = x[2*i+1] - tempi;
                x[2*i] += tempr;
                x[2*i+1] += tempi;
            }
            wtemp = wr;
            wr = wr * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }

    // Scaling for inverse transform
    if (isign == 1) {
        float scale = 1.0 / n;
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            x[2*i] *= scale;
            x[2*i+1] *= scale;
        }
    }
}
