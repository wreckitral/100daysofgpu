#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define BLUR_SIZE 3

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }

        out[row * w + col] = static_cast<unsigned char>(pixVal / pixels);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, imageSize);
    cudaMalloc(&d_out, imageSize);

    cudaMemcpy(d_in, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    blurKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cv::Mat outputImage(inputImage.size(), inputImage.type());
    cudaMemcpy(outputImage.data, d_out, imageSize, cudaMemcpyDeviceToHost);

    cv::imwrite("blurred_output.jpg", outputImage);
    std::cout << "Blurred image saved as 'blurred_output.jpg'" << std::endl;

    cv::imshow("Input Image", inputImage);
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

