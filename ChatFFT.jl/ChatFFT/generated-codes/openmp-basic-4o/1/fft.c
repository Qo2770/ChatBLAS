
#include "chatfft_openmp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Swap macro for in-place array reordering
#define SWAP(a,b) { float temp=(a);(a)=(b);(b)=temp; }

void chatfft(float *x, int n, int isign) {
    int i, j, m, mmax, step;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;

    // Bit-reverse copy
    int n2 = n << 1;
    j = 0;
    for (i = 0; i < n2; i += 2) {
        if (j > i) {
            SWAP(x[j], x[i]);
            SWAP(x[j + 1], x[i + 1]);
        }
        m = n;
        while (m >= 2 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Danielson-Lanczos algorithm
    for (mmax = 2; n2 > mmax; mmax = step) {
        step = mmax << 1;
        theta = isign * (-2 * M_PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        // Parallelize this loop with OpenMP
        #pragma omp parallel for private(i, j, wr, wi, wtemp, tempr, tempi)
        for (m = 0; m < mmax; m += 2) {
            for (i = m; i < n2; i += step) {
                j = i + mmax;
                tempr = wr * x[j] - wi * x[j + 1];
                tempi = wr * x[j + 1] + wi * x[j];
                x[j] = x[i] - tempr;
                x[j + 1] = x[i + 1] - tempi;
                x[i] += tempr;
                x[i + 1] += tempi;
            }
            wtemp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + wtemp * wpi;
        }
    }

    // Inverse FFT normalization
    if (isign == 1) {
        #pragma omp parallel for
        for (i = 0; i < n2; i++) {
            x[i] /= n;
        }
    }
}
