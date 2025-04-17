
#include "chatfft_openmp.h"
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

void chatfft(float *x, int n, int isign) {
    int i, j, k, m, mmax, istep;
    float theta, wtemp, wpr, wpi, wr, wi, tempr, tempi;

    // Bit-reverse the input array
    j = 0;
    for (i = 0; i < n; i += 2) {
        if (j > i) {
            // Swap real parts
            tempr = x[j];
            x[j] = x[i];
            x[i] = tempr;
            // Swap imaginary parts
            tempr = x[j+1];
            x[j+1] = x[i+1];
            x[i+1] = tempr;
        }
        m = n >> 1;
        while (m >= 2 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Danielson-Lanczos section
    mmax = 2;
    while (n > mmax) {
        istep = mmax << 1;
        theta = isign * (2 * PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        #pragma omp parallel for private(k, tempr, tempi, j)
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

    // Normalize if it's the inverse FFT
    if (isign == 1) {
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            x[i] /= n >> 1;
        }
    }
}
