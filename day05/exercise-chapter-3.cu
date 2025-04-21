#include <iostream>

#define WIDTH 4

// 1a. MatMul kernel but each thread produce one matrix row
__global__
void MatrixMulOneThreadPerRow(float* M, float* N, float* P, int Width) {
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

// 1b. MatMul kernel but each thread produce one matrix col
__global__
void MatrixMulOneThreadPerCol(float *M, float *N, float *P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < Width) {
        for (int row = 0; row < Width; ++row) {
            float sum = 0.0f;
            for (int i = 0; i < Width; ++i) {
                sum += M[row * Width + i] * N[i * Width + col];
            }
            P[row * Width + col] = sum;
        }
    }
}

//2. Matrix-vector multiplication
__global__
void MatVecMul(float *M, float *V, float *A, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < Width) {
        float val = 0.0f;

        for (int col = 0; col < Width; col++) {
            val += M[row * Width + col] * V[col];
        }

        A[row] = val;
    }
}

int main() {
    int size = WIDTH * WIDTH * sizeof(float);
    float h_M[WIDTH * WIDTH], h_N[WIDTH * WIDTH], h_P[WIDTH * WIDTH], h_V[WIDTH], h_A[WIDTH];

    // initialize matrices with some test values
    for (int i = 0; i < WIDTH * WIDTH; ++i) {
        h_M[i] = 1.0f;
        h_N[i] = 1.0f;
    }

    for (int i = 0; i < WIDTH; ++i) {
        h_V[i] = 1.0f;
    }

    float *d_M, *d_N, *d_P, *d_V, *d_A;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);
    cudaMalloc(&d_V, sizeof(float) * WIDTH);
    cudaMalloc(&d_A, sizeof(float) * WIDTH);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(float) * WIDTH, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (WIDTH + threadsPerBlock - 1) / threadsPerBlock;
    MatrixMulOneThreadPerRow<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, WIDTH);

    MatVecMul<<<1, 4>>>(d_M, d_V, d_A, WIDTH);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, sizeof(float) * WIDTH, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix P:\n";
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            std::cout << h_P[i * WIDTH + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Result matrix P:\n";
    for (int i = 0; i < WIDTH; ++i) {
        std::cout << h_A[i] << " ";
        std::cout << "\n";
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_V);

    return 0;
}

// 3a. 16 * 32
// 3b. ((N - 1) / 16 + 1) * ((M - 1) / 32 + 1) * 16 * 32 = 48640
// 3c. ((N - 1) / 16 + 1) * ((M - 1) / 32 + 1)
// 3d. N * M


// 4a. 20 * 400 + 10
// 4b. 10 * 500 + 20

// 5a. z * (H * W) + y * W + x
