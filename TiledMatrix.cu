
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixMul(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0;

    for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; tileIdx++) {
        if (row < M && tileIdx * TILE_SIZE + threadIdx.x < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + tileIdx * TILE_SIZE + threadIdx.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && tileIdx * TILE_SIZE + threadIdx.y < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    
    int M = 10000;
    int N = 30000;
    int K = 20000;

    // Allocate memory on the host for input and output matrices
    float *A, *B, *C;
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));

    // Initialize input matrices with random values
    for (int i = 0; i < M * K; i++) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = rand() / (float)RAND_MAX;

    // Allocate memory on the device for input and output matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // Copy input matrices from host to device memory
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    // ... (previous code)

      // Create and record events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Set block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel on the device
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Record and synchronize events
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy output matrix from device to host memory
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // printf("Matrix C:\n");
   // printMatrix(C, M, N);
    printf("Execution Time: %f ms\n", elapsedTime);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
