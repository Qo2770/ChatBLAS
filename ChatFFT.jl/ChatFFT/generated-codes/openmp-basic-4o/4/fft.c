
#include "chatfft_openmp.h"
#include <math.h>
#include <omp.h>

// Helper function to reverse the bits
unsigned int reverse_bits(unsigned int x, int n) {
    unsigned int result = 0;
    for (int i = 0; i < n; i++) {
        if (x & (1U << i))
            result |= 1U << (n - 1 - i);
    }
    return result;
}

// FFT implementation with OpenMP
void chatfft(float *x, int n, int isigm) {
    if (n <= 1) return;

    // Log2n to determine the number of stages
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;

    // Bit reversal reordering
    for (unsigned int i = 0; i < n; i++) {
        unsigned int j = reverse_bits(i, log2n);
        if (j > i) {
            // Swap real and imaginary parts
            float temp_real = x[2 * i];
            float temp_imag = x[2 * i + 1];
            x[2 * i] = x[2 * j];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j] = temp_real;
            x[2 * j + 1] = temp_imag;
        }
    }

    // Cooley-Tukey FFT
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s; // 2^s
        int m2 = m >> 1; // m / 2

        float theta = isigm * (2.0 * M_PI / m);
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);

        #pragma omp parallel for
        for (int j = 0; j < m2; j++) {
            float wr = cos(j * theta);
            float wi = sin(j * theta);

            for (int k = j; k < n; k += m) {
                int t = k + m2;
                float tr = wr * x[2 * t] - wi * x[2 * t + 1];
                float ti = wr * x[2 * t + 1] + wi * x[2 * t];

                // Real and imaginary parts
                float ur = x[2 * k];
                float ui = x[2 * k + 1];

                // Update calculated value
                x[2 * k] = ur + tr;
                x[2 * k + 1] = ui + ti;
                x[2 * t] = ur - tr;
                x[2 * t + 1] = ui - ti;
            }
        }
    }

    // If inverse FFT, divide by n
    if (isigm == 1) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[2 * i] /= n;
            x[2 * i + 1] /= n;
        }
    }
}
