
#include <math.h>
#include <omp.h>
#include "chatfft_openmp.h"

void chatfft(float *x, int n, int isign) {
    int i, j, m, mmax, step;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;

    // Bit-reversal permutation
    j = 0;
    for (i = 0; i < n; i += 2) {
        if (j > i) {
            // Swap the real part
            float temp = x[j];
            x[j] = x[i];
            x[i] = temp;

            // Swap the imaginary part
            temp = x[j + 1];
            x[j + 1] = x[i + 1];
            x[i + 1] = temp;
        }
        int m = n >> 1;
        while (m >= 2 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Danielson-Lanczos section
    for (mmax = 2; n > mmax; mmax <<= 1) {
        theta = isign * (2 * M_PI / mmax);
        wpr = (float)(cos(theta) - 1.0);
        wpi = (float)sin(theta);
        wr = 1.0;
        wi = 0.0;
        
        step = mmax << 1;

        #pragma omp parallel for private(wr, wi, j, wtemp, tempr, tempi)
        for (m = 0; m < mmax; m += 2) {
            for (i = m; i < n; i += step) {
                j = i + mmax;
                tempr = wr * x[j] - wi * x[j + 1];
                tempi = wr * x[j + 1] + wi * x[j];
                x[j] = x[i] - tempr;
                x[j + 1] = x[i + 1] - tempi;
                x[i] += tempr;
                x[i + 1] += tempi;
            }
            // Trigonometric recurrence
            wtemp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + wtemp * wpi;
        }
    }

    // Normalize if inverse FFT
    if (isign == 1) {
        float scale = 1.0 / n;
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] *= scale;
        }
    }
}
