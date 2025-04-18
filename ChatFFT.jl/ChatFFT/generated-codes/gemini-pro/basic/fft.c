#include "chatfft_openmp.h" // Include the requested header

#define _USE_MATH_DEFINES   // Define this before math.h for M_PI on some systems (like MSVC)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Include OpenMP header

// Helper function to swap two complex numbers (represented by 4 floats)
static inline void swap_complex(float *x, int i, int j) {
    // Indices for real and imaginary parts
    int real_i = 2 * i;
    int imag_i = real_i + 1;
    int real_j = 2 * j;
    int imag_j = real_j + 1;

    float temp_real = x[real_i];
    float temp_imag = x[imag_i];
    x[real_i] = x[real_j];
    x[imag_i] = x[imag_j];
    x[real_j] = temp_real;
    x[imag_j] = temp_imag;
}

/**
 * @brief Computes the Fast Fourier Transform (FFT) or its inverse in-place.
 *
 * This function implements the Cooley-Tukey radix-2 FFT algorithm.
 * The input vector x is modified directly.
 * OpenMP is used to parallelize the butterfly computation stages.
 *
 * @param x Pointer to the input/output vector of complex numbers.
 *          Stored as interleaved floats: [Re(0), Im(0), Re(1), Im(1), ...].
 *          The total size of this float array must be 2*n.
 * @param n The number of complex samples in the vector (must be a power of 2).
 * @param isign Sign determining the transform direction:
 *              -1 for forward FFT (DFT)
 *              +1 for inverse FFT (IDFT). Note: IDFT result is automatically scaled by 1/n.
 */
void chatfft(float *x, int n, int isign) {
    // --- Basic Checks ---
    if (x == NULL) {
        fprintf(stderr, "Error (chatfft): Input array x is NULL.\n");
        return;
    }
    if (n <= 0) {
         fprintf(stderr, "Error (chatfft): FFT size n=%d must be positive.\n", n);
         return;
    }
    // Check if n is a power of 2
    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "Error (chatfft): FFT size n=%d must be a power of 2.\n", n);
        return; // Radix-2 FFT requires power-of-2 size
    }
    if (isign != 1 && isign != -1) {
        fprintf(stderr, "Error (chatfft): isign must be 1 or -1.\n");
        return;
    }

    // --- 1. Bit-Reversal Permutation ---
    // Rearrange the input data according to bit-reversed indices
    // This is done serially as parallelizing swaps can be complex and often
    // doesn't provide significant speedup compared to the butterfly stages.
    int log2n = 0;
    while ((1 << log2n) < n) {
        log2n++;
    }

    for (int i = 0; i < n; i++) {
        int j = 0;
        // Calculate bit-reversed index j for i
        for (int k = 0; k < log2n; k++) {
            if ((i >> k) & 1) { // If the k-th bit of i is 1
                j |= (1 << (log2n - 1 - k)); // Set the (log2n-1-k)-th bit of j
            }
        }

        // Swap elements i and j only if j > i to avoid double swaps
        if (j > i) {
            swap_complex(x, i, j);
        }
    }

    // --- 2. Cooley-Tukey Butterfly Stages ---
    // Iterate through stages (s = 1 to log2n)
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;    // Size of the FFT blocks at this stage (2, 4, 8, ...)
        int m2 = m >> 1;   // Half size (m/2)
        double theta_base = (isign * M_PI) / m2; // Base angle increment for twiddle factors

        // Parallelize the computation of butterflies within this stage
        // Each thread handles a subset of the 'j' loop iterations (start indices of FFT blocks)
        #pragma omp parallel for shared(x, n, m, m2, theta_base, isign) default(none) schedule(static)
        for (int j = 0; j < n; j += m) {
            // Calculate twiddle factors W_m^k = exp(isign * 2*pi*i * k / m)
            // Pre-calculate Wr=1, Wi=0 for k=0
            float wr = 1.0f;
            float wi = 0.0f;

            // Iterate through pairs within the FFT block
            for (int k = 0; k < m2; k++) {
                // Indices for the pair of elements to combine (x[j+k] and x[j+k+m2])
                int idx1_real = 2 * (j + k);
                int idx1_imag = idx1_real + 1;
                int idx2_real = 2 * (j + k + m2);
                int idx2_imag = idx2_real + 1;

                // Complex multiplication: temp = W * x[idx2]
                // temp_real = Wr * x[idx2_real] - Wi * x[idx2_imag]
                // temp_imag = Wr * x[idx2_imag] + Wi * x[idx2_real]
                float tr = wr * x[idx2_real] - wi * x[idx2_imag];
                float ti = wr * x[idx2_imag] + wi * x[idx2_real];

                // Butterfly calculation:
                // x[idx2] = x[idx1] - temp
                // x[idx1] = x[idx1] + temp
                float ur = x[idx1_real]; // Store x[idx1] temporarily
                float ui = x[idx1_imag];

                x[idx1_real] = ur + tr;
                x[idx1_imag] = ui + ti;
                x[idx2_real] = ur - tr;
                x[idx2_imag] = ui - ti;

                // Update twiddle factor for next k using trigonometric recurrence
                // W_m^(k+1) = W_m^k * W_m^1
                // This is usually faster than calling sin/cos repeatedly
                if (k < m2 -1) { // Avoid calculation for the last k
                     // Calculate W_m^1 = cos(theta_base) + i*sin(theta_base) if not done yet for this j-block
                     // But simpler (though slightly less efficient) is direct calculation per k:
                     double theta = theta_base * (k + 1); // Angle for W_m^(k+1)
                     wr = (float)cos(theta);
                     wi = (float)sin(theta);
                     // Note: For higher performance, one could use the recurrence relation
                     // float w_temp = wr;
                     // wr = wr * w_real_base - wi * w_imag_base; // w_real_base = cos(theta_base)
                     // wi = w_temp * w_imag_base + wi * w_real_base; // w_imag_base = sin(theta_base)
                     // Requires calculating w_real_base and w_imag_base once per 'j' loop or even per 's' stage.
                }
            }
        } // end parallel for j
    } // end for s (stages)


    // --- 3. Inverse FFT Scaling ---
    // If computing the inverse FFT (isign=1), scale the result by 1/n
    if (isign == 1) {
        float scale_factor = 1.0f / (float)n;
        // Parallelize the scaling loop
        #pragma omp parallel for shared(x, n, scale_factor) default(none) schedule(static)
        for (int i = 0; i < 2 * n; i++) { // Scale both real and imaginary parts
            x[i] *= scale_factor;
        }
    }
}
