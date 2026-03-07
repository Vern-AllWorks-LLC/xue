/*
 * _xue_cuda_kernels.cu — CUDA GPU kernels for xue tensor operations
 *
 * Standard parallel elementwise and tiled matmul kernels.
 * Compiled separately, loaded at runtime via dlopen.
 *
 * Build: nvcc -shared -O3 -o _xue_cuda_kernels.so _xue_cuda_kernels.cu -lcudart
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define TILE_DIM 16

/* ── Elementwise kernels ──────────────────────────────────────────── */

__global__ void kernel_add(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_sub(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void kernel_mul(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_div(double *out, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}

__global__ void kernel_scalar_mul(double *out, const double *a, double s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}

__global__ void kernel_neg(double *out, const double *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

/* ── Tiled matrix multiply ────────────────────────────────────────── */

__global__ void kernel_matmul(double *C, const double *A, const double *B,
                               int M, int K, int N) {
    __shared__ double sA[TILE_DIM][TILE_DIM];
    __shared__ double sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_row = t * TILE_DIM + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

/* ── Host-callable wrappers (extern "C" for dlopen) ───────────────── */

extern "C" {

void cuda_elementwise_add(double *out, const double *a, const double *b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    cudaDeviceSynchronize();
}

void cuda_elementwise_sub(double *out, const double *a, const double *b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sub<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    cudaDeviceSynchronize();
}

void cuda_elementwise_mul(double *out, const double *a, const double *b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_mul<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    cudaDeviceSynchronize();
}

void cuda_elementwise_div(double *out, const double *a, const double *b, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_div<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    cudaDeviceSynchronize();
}

void cuda_scalar_mul(double *out, const double *a, double s, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_scalar_mul<<<blocks, BLOCK_SIZE>>>(out, a, s, n);
    cudaDeviceSynchronize();
}

void cuda_matmul(double *C, const double *A, const double *B, int M, int K, int N) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    kernel_matmul<<<blocks, threads>>>(C, A, B, M, K, N);
    cudaDeviceSynchronize();
}

}  /* extern "C" */
