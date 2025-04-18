#include "chatfft_openmp.h" // As requested

#include <math.h>
#include <stdio.h> // For potential error checking/debugging (optional)
#include <stdlib.h> // For exit() (optional)
#include <omp.h>   // For OpenMP directives and functions

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//-----------------------------------------------------------------------------
// Helper function: Calculates log2(n)
//-----------------------------------------------------------------------------
static inline int ilog2(int n) {
    int log_n = 0;
    // Assume n is a power of 2, n >= 1
    while ((1 << log_n) < n) {
        log_n++;
    }
    // Check if n is actually a power of 2 (optional but good practice)
    if ((1 << log_n) != n && n != 0) {
         // This should not happen based on the problem description assumption
         fprintf(stderr, "Error: n (%d) is not a power of 2 in ilog2.\n", n);
         exit(EXIT_FAILURE);
    }
    return log_n;
}

//-----------------------------------------------------------------------------
// Helper function: Performs bit reversal of an integer
// Assumes n is a power of 2, log_n = log2(n)
//-----------------------------------------------------------------------------
static inline int bit_reverse(int j, int log_n) {
    int reversed_j = 0;
    for (int i = 0; i < log_n; ++i) {
        if ((j >> i) & 1) { // Check if the i-th bit of j is set
            reversed_j |= 1 << (log_n - 1 - i);
        }
    }
    return reversed_j;
}

//-----------------------------------------------------------------------------
// chatfft: Computes in-place complex-to-complex FFT
// x:     Pointer to float array [real0, imag0, real1, imag1, ...]
// n:     Number of complex points (must be a power of 2). Array x has 2*n elements.
// isign: -1 for forward FFT, 1 for inverse FFT
//-----------------------------------------------------------------------------
void chatfft(float *x, int n, int isign) {

    if (n <= 1) {
        // FFT of size 0 or 1 is trivial (identity)
        // Inverse FFT of size 1 might need normalization if defined that way,
        // but typically division by n=1 changes nothing.
        return;
    }

    // --- Parameter Checks (Optional but Recommended) ---
    if (x == NULL) {
        fprintf(stderr, "Error: Input array x is NULL.\n");
        return; // Or exit
    }
    if (n < 1 || (n & (n - 1)) != 0) {
         fprintf(stderr, "Error: n (%d) must be a positive power of 2.\n", n);
         return; // Or exit
    }
    if (isign != 1 && isign != -1) {
         fprintf(stderr, "Error: isign (%d) must be 1 or -1.\n", isign);
         return; // Or exit
    }

    // --- Calculate log2(n) ---
    const int log_n = ilog2(n);

    // --- 1. Bit Reversal Permutation ---
    // Rearrange the input data according to bit-reversed indices.
    // This is often sequential as parallelizing swaps safely can be tricky
    // or introduce overhead. However, for large N, parallelizing might offer
    // benefits if done carefully. The check 'if (i > j)' prevents swapping twice
    // and makes simple parallelization potentially safe from race conditions
    // on the *pair* being swapped (though contention on cache lines is possible).
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
        int i = bit_reverse(j, log_n);

        // Swap elements only if i > j to avoid swapping twice
        if (i > j) {
            // Indices in the float array (interleaved complex)
            int idx_j_real = 2 * j;
            int idx_j_imag = idx_j_real + 1;
            int idx_i_real = 2 * i;
            int idx_i_imag = idx_i_real + 1;

            // Swap real parts
            float temp_real = x[idx_j_real];
            x[idx_j_real] = x[idx_i_real];
            x[idx_i_real] = temp_real;

            // Swap imaginary parts
            float temp_imag = x[idx_j_imag];
            x[idx_j_imag] = x[idx_i_imag];
            x[idx_i_imag] = temp_imag;
        }
    }

    // --- 2. Iterative Butterfly Stages (Radix-2 DIT) ---
    // Loop through stages s = 1 to log_n
    for (int s = 1; s <= log_n; ++s) {
        const int m = 1 << s;         // Butterfly size for this stage (2, 4, 8, ..., n)
        const int m2 = m >> 1;        // Half butterfly size (1, 2, 4, ..., n/2)

        // Calculate the principal m-th root of unity for this stage W_m
        // W_m = exp(-isign * 2 * pi * i / m)
        // Use double for intermediate trigonometric calculation for precision
        const double angle_step = (double)isign * -M_PI / (double)m2; // angle = -isign * 2*pi/m = -isign*pi/m2
        const double wr_step = cos(angle_step); // Real part of W_m step factor
        const double wi_step = sin(angle_step); // Imaginary part of W_m step factor

        // Loop through the butterflies *within* a block (k = 0 to m/2 - 1)
        // This loop calculates the twiddle factor W^k_{m} = exp(-isign * 2*pi*i*k / m)
        // W starts at 1.0 + 0.0i for k=0
        double wpr = 1.0; // Real part of twiddle factor W^k_m
        double wpi = 0.0; // Imaginary part of twiddle factor W^k_m

        for (int k = 0; k < m2; ++k) {
            // This inner loop iterates through the groups of butterflies
            // across the entire data length. This is the main candidate for parallelization.
            // Optimization: Use static scheduling for large N as workload per thread
            //               should be relatively balanced within this loop.
            #pragma omp parallel for schedule(static)
            for (int j = k; j < n; j += m) {
                // Indices for the butterfly operation: j and j + m/2
                int idx1_real = 2 * j;
                int idx1_imag = idx1_real + 1;
                int idx2_real = 2 * (j + m2);
                int idx2_imag = idx2_real + 1;

                // Get the two complex numbers: t1 = x[j], t2 = x[j+m/2]
                float t1_real = x[idx1_real];
                float t1_imag = x[idx1_imag];
                float t2_real = x[idx2_real];
                float t2_imag = x[idx2_imag];

                // Calculate the product: temp = W * t2
                // where W = wpr + i * wpi (current twiddle factor)
                // Use float for the actual computation on the data
                float temp_real = (float)wpr * t2_real - (float)wpi * t2_imag;
                float temp_imag = (float)wpr * t2_imag + (float)wpi * t2_real;

                // Calculate the butterfly outputs:
                // x[j]      = t1 + temp
                // x[j+m/2]  = t1 - temp
                // Store results back in-place
                x[idx1_real] = t1_real + temp_real;
                x[idx1_imag] = t1_imag + temp_imag;
                x[idx2_real] = t1_real - temp_real;
                x[idx2_imag] = t1_imag - temp_imag;
            }

            // Update the twiddle factor W for the next k using recurrence:
            // W_{k+1} = W_k * W_m_step
            // (wpr + i*wpi) * (wr_step + i*wi_step)
            // Need temporary storage to avoid using updated wpr in wpi calculation
            double wpr_next = wpr * wr_step - wpi * wi_step;
            double wpi_next = wpr * wi_step + wpi * wr_step;
            wpr = wpr_next;
            wpi = wpi_next;
        }
    }

    // --- 3. Normalization for Inverse FFT ---
    // If performing an inverse FFT (isign=1), divide all elements by n.
    if (isign == 1) {
        const float norm_factor = 1.0f / (float)n;
        // Parallelize the normalization loop
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < 2 * n; ++i) {
            x[i] *= norm_factor;
        }
    }
}
