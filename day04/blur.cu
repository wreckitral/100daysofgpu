#include <opencv2/opencv.hpp>
#include <iostream>

#define BLUR_SIZE 3

void cpuBlurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int pixVal = 0;
            int pixels = 0;

            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        pixVal += in[curRow * w + curCol];
                        ++pixels;
                    }
                }
            }

            out[row * w + col] = (unsigned char)(pixVal / pixels);
        }
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

    cv::Mat outputImage(inputImage.size(), inputImage.type());

    cpuBlurKernel(inputImage.data, outputImage.data, inputImage.cols, inputImage.rows);

    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Image", inputImage);

    cv::namedWindow("Blurred Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blurred Image", outputImage);

    cv::imwrite("blurred_output.jpg", outputImage);
    std::cout << "Blurred image saved as 'blurred_output.jpg'" << std::endl;

    cv::waitKey(0);

    return 0;
}
