
#include "chatfft_openmp.h"
#include <math.h>
#include <omp.h>
#include <complex.h>

#define PI 3.14159265358979323846

void chatfft(float *x, int n, int isigm) {
    int i, j, m, m1, m2, k;
    int num_subarrays;
    float c, s, t_real, t_imag, u_real, u_imag;
    int stage, size, halfsize, tablestep;
    
    // Ensure x is interpreted as complex numbers
    float (*cx)[2] = (float (*)[2])x;

    // Bit reversal sorting
    j = 0;
    for (i = 0; i < n/2; i++) {
        if (i < j) {
            // Swap two complex elements
            float temp_real = cx[i][0];
            float temp_imag = cx[i][1];
            cx[i][0] = cx[j][0];
            cx[i][1] = cx[j][1];
            cx[j][0] = temp_real;
            cx[j][1] = temp_imag;
        }
        m = n / 2;
        while (j >= m && m >= 2) {
            j -= m;
            m /= 2;
        }
        j += m;
    }
    
    // FFT computation using OpenMP parallelization
    num_subarrays = 1;
    tablestep = n;

    for (stage = 0; stage < log2(n); stage++) {
        size = 2 * num_subarrays;
        halfsize = num_subarrays;
        tablestep /= 2;
        double angle = isigm * PI / halfsize;

        // Parallel loop over each subarray
        #pragma omp parallel for private(i, k, c, s, t_real, t_imag, u_real, u_imag)
        for (j = 0; j < num_subarrays; j++) {
            c = cos(j * angle);
            s = sin(j * angle);
            for (i = j; i < n/2; i += size) {
                k = i + halfsize;
                t_real = c * cx[k][0] - s * cx[k][1];
                t_imag = s * cx[k][0] + c * cx[k][1];
                u_real = cx[i][0];
                u_imag = cx[i][1];
                cx[i][0] = u_real + t_real;
                cx[i][1] = u_imag + t_imag;
                cx[k][0] = u_real - t_real;
                cx[k][1] = u_imag - t_imag;
            }
        }
        num_subarrays = size;
    }

    // Normalize if it's an inverse FFT
    if (isigm == 1) {
        float norm_factor = 1.0f / n;
        #pragma omp parallel for private(i)
        for (i = 0; i < n/2; i++) {
            cx[i][0] *= norm_factor;
            cx[i][1] *= norm_factor;
        }
    }
}
