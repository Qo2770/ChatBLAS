
#include "chatfft_openmp.h"
#include <omp.h>
#include <math.h>

#define SWAP(a, b) { float temp = (a); (a) = (b); (b) = temp; }

void chatfft(float *x, int n, int isign) {
    int i, j, m, mmax, istep;
    double wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;

    // Bit-reverse the input data for in-place radix-2 FFT
    j = 0;
    for (i = 0; i < n; i += 2) {
        if (j > i) {
            SWAP(x[j], x[i]);
            SWAP(x[j+1], x[i+1]);
        }
        m = n;
        while (m >= 2 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey FFT algorithm
    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = isign * (2 * M_PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        // Parallelization with OpenMP
        #pragma omp parallel for private(i, j, tempr, tempi)
        for (m = 0; m < mmax; m += 2) {
            for (i = m; i < n; i += istep) {
                j = i + mmax;
                tempr = wr * x[j] - wi * x[j+1];
                tempi = wr * x[j+1] + wi * x[j];
                x[j] = x[i] - tempr;
                x[j+1] = x[i+1] - tempi;
                x[i] += tempr;
                x[i+1] += tempi;
            }
            wtemp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + wtemp * wpi;
        }
        mmax = istep;
    }

    // Scaling for the inverse transform
    if (isign == 1) {
        #pragma omp parallel for private(i)
        for (i = 0; i < n; i++) {
            x[i] /= n / 2;
        }
    }
}
