#include <cuda.h>
#include <stdio.h>

// define a struct for matrix
typedef struct {
    int rows;
    int cols;
    float* elements;

} Matrix;

// forward helper function declaration
void populateMatrix(Matrix* m, int rows, int cols);
void freeMatrix(Matrix *m);

// thread-block size
#define BLOCK_SIZE 16

// matrix multiplication kernel
// assuming that the matrices dimensions are multiples of BLOCK_SIZE
__global__ void MatMul(Matrix a, Matrix b, Matrix c)
{
    float sum = 0; // initiate sum's default value

    // col and row was get from the block's index * block's dimension + thread's id
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // each thread produce only one element of c according to what col and row that value is
    for (int i = 0; i < a.cols ; i++) {
        sum += a.elements[row * a.cols + i] * b.elements[i * b.cols + col];
        c.elements[row * c.cols + col] = sum;
    }
}

int main()
{
    Matrix a, b, c; // declare matrices on host
    Matrix d_a, d_b, d_c; // declare matrices on device

    populateMatrix(&a, 64, 64); // populate matrix a with random number
    populateMatrix(&b, 64, 64); // populate matrix b with random number
    populateMatrix(&c, 64, 64); // populate matrix b with random number

    size_t size = a.cols * a.rows * sizeof(float); // declare size for matrix d_a (device)
    cudaMalloc(&d_a.elements, size); // allocate memory for matrix d_a (device)
    cudaMemcpy(d_a.elements, a.elements, size, cudaMemcpyHostToDevice); // copy matrix a (host) to the allocated memory on d_a (device)

    // do the same thing for matrix b
    size = b.cols * b.rows * sizeof(float);
    cudaMalloc(&d_b.elements, size);
    cudaMemcpy(d_b.elements, b.elements, size, cudaMemcpyHostToDevice);

    // matrix c = matrix a * matrix b,
    // that means (N x M) * (M x Q) = (N x Q)
    size = a.rows * b.cols * sizeof(float);
    cudaMalloc(&d_c.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // define dimension of each block in the grid
    dim3 dimGrid(b.cols / dimBlock.x, a.rows / dimBlock.y); // define the dimension of the grid of blocks

    // launch kernel
    MatMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    // copy result to Matrix c (host)
    cudaMemcpy(c.elements, d_c.elements, size, cudaMemcpyDeviceToHost);

    // free on host
    freeMatrix(&a);
    freeMatrix(&b);

    // free memory on device
    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    cudaFree(d_c.elements);
}

void populateMatrix(Matrix* m, int rows, int cols)
{
    m->rows = rows;
    m->cols = cols;
    int size = rows * cols * sizeof(float);

    m->elements = (float*)malloc(size);

    for(int i = 0; i < rows * cols; i++)
    {
        m->elements[i] = (float)rand() / RAND_MAX;
    }
}

void freeMatrix(Matrix *m) {
    free(m->elements);
}
