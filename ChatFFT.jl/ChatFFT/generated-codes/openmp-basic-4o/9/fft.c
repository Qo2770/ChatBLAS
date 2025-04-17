
#include "chatfft_openmp.h"
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

void chatfft(float *x, int n, int isign) {
    // Bit-reversal permutation
    int i, j, k, m;
    for (i = 1, j = 0; i < (n << 1); i += 2) {
        if (j > i) {
            float tmp_real = x[j];
            float tmp_imag = x[j + 1];
            x[j] = x[i];
            x[j + 1] = x[i + 1];
            x[i] = tmp_real;
            x[i + 1] = tmp_imag;
        }
        m = n;
        while (m >= 2 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey FFT
    int step;
    for (step = 2; step <= (n << 1); step <<= 1) {
        double delta = isign * (2.0 * PI / step);
        double sine = sin(delta / 2.0);
        double factor_real = -2.0 * sine * sine;
        double factor_imag = sin(delta);

        #pragma omp parallel for private(i, m, j)
        for (i = 0; i < n << 1; i += step) {
            double w_real = 1.0;
            double w_imag = 0.0;
            for (m = 0; m < step; m += 2) {
                j = i + m;

                double temp_real = w_real * x[j + step] - w_imag * x[j + step + 1];
                double temp_imag = w_real * x[j + step + 1] + w_imag * x[j + step];
                
                x[j + step] = x[j] - temp_real;
                x[j + step + 1] = x[j + 1] - temp_imag;

                x[j] = x[j] + temp_real;
                x[j + 1] = x[j + 1] + temp_imag;

                double temp_wr = w_real;
                w_real += temp_wr * factor_real - w_imag * factor_imag;
                w_imag += w_imag * factor_real + temp_wr * factor_imag;
            }
        }
    }

    // Scaling for inverse transform
    if (isign == 1) {
        float scale = 1.0 / n;
        #pragma omp parallel for
        for (i = 0; i < (n << 1); i++) {
            x[i] *= scale;
        }
    }
}
