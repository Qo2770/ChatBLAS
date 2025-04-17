
#include "chatfft_openmp.h"
#include <math.h>
#include <omp.h>

void chatfft(float *x, int n, int isign) {
    int i, j, k, m, mmax, istep;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;

    // Bit-reverse the input data
    j = 0;
    for (i = 0; i < n; i += 2) {
        if (j > i) {
            // Swap the real parts
            float temp_real = x[j];
            x[j] = x[i];
            x[i] = temp_real;
            
            // Swap the imaginary parts
            float temp_imag = x[j + 1];
            x[j + 1] = x[i + 1];
            x[i + 1] = temp_imag;
        }
        m = n / 2;
        while (m >= 2 && j >= m) {
            j -= m;
            m /= 2;
        }
        j += m;
    }

    // Danielson-Lanczos section
    mmax = 2;
    while (n > mmax) {
        istep = 2 * mmax;
        theta = isign * (2 * M_PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        
        #pragma omp parallel for num_threads(omp_get_max_threads()) private(i, j, k, tempr, tempi)
        for (m = 0; m < mmax; m += 2) {
            for (i = m; i < n; i += istep) {
                j = i + mmax;
                tempr = wr * x[j] - wi * x[j + 1];
                tempi = wr * x[j + 1] + wi * x[j];
                
                x[j] = x[i] - tempr;
                x[j + 1] = x[i + 1] - tempi;
                
                x[i] += tempr;
                x[i + 1] += tempi;
            }
            wtemp = wr;
            wr = wr * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        
        mmax = istep;
    }
    
    // For the inverse transform, scale the output by 1/n
    if (isign == 1) {
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (i = 0; i < n; i++) {
            x[i] /= n;
        }
    }
}
