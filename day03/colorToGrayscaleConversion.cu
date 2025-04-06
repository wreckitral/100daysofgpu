#include <cstddef>
#include <cstdio>
#include <opencv2/opencv.hpp>
using namespace cv;
#define CHANNELS 3

__global__
void conversionKernel(uchar *Pout, uchar *Pin, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        int grayOffset = row * width + col; // 1D offset for grayscale
        int rgbOffset = grayOffset * CHANNELS; // times CHANNELS because rgb use 3 bytes
        uchar r = Pin[rgbOffset];
        uchar g = Pin[rgbOffset + 1];
        uchar b = Pin[rgbOffset + 2];
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void loadimage(const std::string &filename, uchar* &input_h, int &width, int &height, size_t &totalpixel, size_t &graypixel);
void outputimage(int &width, int &height, uchar* &output_h);

int main()
{
    uchar *input_h, *input_d, *output_d;
    int width, height;
    size_t totalpixel, graypixel;

    loadimage("depa.jpeg", input_h, width, height, totalpixel, graypixel);

    cudaMalloc(&input_d, totalpixel);
    cudaMalloc(&output_d, graypixel);

    cudaMemcpy(input_d, input_h, totalpixel, cudaMemcpyHostToDevice);

    dim3 gridSize(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 blockSize(16, 16);

    conversionKernel<<<gridSize, blockSize>>>(output_d, input_d, width, height);
    cudaDeviceSynchronize();

    uchar* output_h = new uchar[graypixel];
    cudaMemcpy(output_h, output_d, graypixel, cudaMemcpyDeviceToHost);

    outputimage(width, height, output_h);

    cudaFree(output_d);
    cudaFree(input_d);
    delete[] output_h;

    return 0;
}

void loadimage(const std::string &filename, uchar* &input_h, int &width, int &height, size_t &totalpixel, size_t &graypixel)
{
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Image not found!\n";
        exit(1);  // Return false doesn't work in a void function
    }
    width = image.cols;
    height = image.rows;
    totalpixel = width * height * CHANNELS; // total pixel
    graypixel = width * height; // gray pixel

    input_h = new uchar[totalpixel];
    memcpy(input_h, image.data, totalpixel);
}

void outputimage(int &width, int &height, uchar* &output_h)
{
    Mat grayImage(height, width, CV_8UC1, output_h);
    imwrite("output.jpeg", grayImage);
}
