// 1a. MatMul kernel but each thread produce one matrix row
#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 4

__global__ void MatrixMulOneThreadPerRow(float* M, float* N, float* P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < Width; ++k) {
                sum += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = sum;
        }
    }
}

int main() {
    int size = WIDTH * WIDTH * sizeof(float);
    float h_M[WIDTH * WIDTH], h_N[WIDTH * WIDTH], h_P[WIDTH * WIDTH];

    // initialize matrices with some test values
    for (int i = 0; i < WIDTH * WIDTH; ++i) {
        h_M[i] = 1.0f;
        h_N[i] = 1.0f;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (WIDTH + threadsPerBlock - 1) / threadsPerBlock;
    MatrixMulOneThreadPerRow<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, WIDTH);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix P:\n";
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            std::cout << h_P[i * WIDTH + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}