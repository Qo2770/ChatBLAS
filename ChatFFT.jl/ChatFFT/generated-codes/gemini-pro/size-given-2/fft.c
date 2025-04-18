#include "chatfft_openmp.h" // As requested
#include <math.h>          // For sinf, cosf, log2f, M_PI
#include <omp.h>           // For OpenMP directives and functions
#include <stdlib.h>        // For malloc/free (if precomputing twiddles, not strictly needed here)
#include <stdio.h>         // For potential error/debug output

// Define PI if not provided by math.h (common practice)
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/**
 * @brief Performs an in-place complex-to-complex Fast Fourier Transform (FFT).
 *
 * This function computes the FFT or inverse FFT of a complex vector `x` using
 * the Cooley-Tukey radix-2 decimation-in-time algorithm.
 * The computation is done in-place, modifying the input vector `x`.
 * OpenMP is used to parallelize the butterfly computations.
 *
 * @param x Pointer to the complex vector data, stored interleaved (real0, imag0, real1, imag1, ...).
 *          The array must have 2*n elements.
 * @param n The number of complex elements in the vector (length of the transform).
 *          MUST be a power of 2.
 * @param isign Sign indicator for the transform direction:
 *              -1 for Forward FFT (DFT)
 *               1 for Inverse FFT (IDFT)
 */
void chatfft(float *x, int n, int isign) {
    // --- Input Validation ---
    if (n <= 0) {
        // fprintf(stderr, "Error: FFT size n must be positive.\n");
        return;
    }
    // Check if n is a power of 2 (simple bit manipulation check)
    if ((n & (n - 1)) != 0) {
       // fprintf(stderr, "Error: FFT size n=%d is not a power of 2.\n", n);
       return; // Or handle error appropriately
    }
    if (n == 1) {
        return; // FFT of size 1 is trivial
    }

    // --- 1. Bit Reversal Permutation ---
    // Rearrange the input elements according to bit-reversed indices.
    // This is necessary for the in-place Cooley-Tukey algorithm.
    int log2n = 0;
    while ((1 << log2n) < n) {
        log2n++;
    }

    // We can parallelize this part, though for n=65536, the butterfly stages
    // will likely dominate the runtime. Ensure proper synchronization or
    // structure to avoid race conditions if parallelized.
    // A common safe parallel approach iterates i from 1 to n-1 and swaps
    // if i < j, ensuring each pair is swapped only once.
    // For simplicity and focus on butterfly parallelization, sequential is shown:
    for (int i = 0; i < n; ++i) {
        int j = 0;
        int temp_i = i;
        // Create the bit-reversed index j
        for (int k = 0; k < log2n; ++k) {
            j = (j << 1) | (temp_i & 1);
            temp_i >>= 1;
        }

        // Swap elements i and j if j > i to avoid double swaps
        if (j > i) {
            // Swap real parts
            float temp_real = x[2 * i];
            x[2 * i] = x[2 * j];
            x[2 * j] = temp_real;
            // Swap imaginary parts
            float temp_imag = x[2 * i + 1];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j + 1] = temp_imag;
        }
    }

    // --- 2. Butterfly Computations ---
    // Perform the Cooley-Tukey DIT butterfly operations stage by stage.
    for (int stage = 1; stage <= log2n; ++stage) {
        int m = 1 << stage;       // Size of FFT blocks at this stage
        int m2 = m >> 1;          // Half size (number of butterflies per block)

        // Twiddle factor calculation setup for this stage
        // W_m = exp(-isign * 2 * pi * i / m)
        // We use the trigonometric recurrence W_m^k = W_m^(k-1) * W_m^1
        // W_m^1 (primitive root):
        float wm_real = cosf(M_PI / m2);
        float wm_imag = (float)isign * sinf(M_PI / m2); // Note: Sign convention depends on FFT definition
                                                        // Standard DFT often uses exp(-j...), so forward (isign=-1) needs +sin
                                                        // Inverse (isign=1) needs -sin
                                                        // Let's adjust based on typical definition:
        wm_imag = -(float)isign * sinf(M_PI / m2);


        // Parallelize the outer loop over butterfly groups (j)
        // Each thread handles independent blocks of data.
        // `static` scheduling is often good for uniform workloads like FFT butterflies.
        // Experiment with `schedule(guided)` or chunk size for potential tuning.
        #pragma omp parallel for schedule(static) // Default chunk size likely okay
        for (int j = 0; j < n; j += m) {
            float w_real = 1.0f; // W_m^0 = 1
            float w_imag = 0.0f;

            // Inner loop iterates through butterflies within a block
            for (int k = 0; k < m2; ++k) {
                int idx1 = 2 * (j + k);          // Index of the first element (real part)
                int idx2 = 2 * (j + k + m2);     // Index of the second element (real part)

                // Read complex numbers a = x[j+k] and b = x[j+k+m2]
                float ar = x[idx1];
                float ai = x[idx1 + 1];
                float br = x[idx2];
                float bi = x[idx2 + 1];

                // Compute the twiddle factor product: T = W * b
                // T_real = w_real * br - w_imag * bi;
                // T_imag = w_real * bi + w_imag * br;
                float T_real = w_real * br - w_imag * bi;
                float T_imag = w_real * bi + w_imag * br;


                // Compute the butterfly operation:
                // x[j+k]     = a + T
                // x[j+k+m2] = a - T
                x[idx1]     = ar + T_real;
                x[idx1 + 1] = ai + T_imag;
                x[idx2]     = ar - T_real;
                x[idx2 + 1] = ai - T_imag;

                // Update twiddle factor W for next butterfly: W = W * W_m
                // Use temporary variables to avoid using updated values prematurely
                float next_w_real = w_real * wm_real - w_imag * wm_imag;
                float next_w_imag = w_real * wm_imag + w_imag * wm_real;
                w_real = next_w_real;
                w_imag = next_w_imag;
            }
        }
         // Implicit barrier at the end of the parallel region ensures all
         // threads finish this stage before proceeding to the next.
    }

    // --- 3. Scaling for Inverse FFT ---
    // If performing an inverse FFT, scale the result by 1/n.
    if (isign == 1) {
        float scale = 1.0f / (float)n;
        // This scaling loop is trivially parallelizable.
        #pragma omp parallel for
        for (int i = 0; i < 2 * n; ++i) { // Iterate through all float values (real and imag)
            x[i] *= scale;
        }
    }
}
