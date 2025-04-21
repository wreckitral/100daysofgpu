# 100 Days of GPU Programming Learning Logs

Documentation, implementation, and codes to my journey of learning GPU programming.

Credit to: https://github.com/hkproj/100-days-of-gpu

Bro in the 100 days challenge: https://github.com/mustafasegf/cuda-100-days-challange

## Day 1

### Summary:
GPU is optimized for handling high performance task, unlike CPU, which focus on
sequential processing, GPUs have thousands of smaller cores designed for parallel excecution,
ideal for model training, simple computation intensive processing, graphics processing.

### Learned:
- GPUs computing
- why is it important for GPU in todays computation
- difference between CPU and GPU design
- why we need parallel computing
- learn basic CUDA with vector addition problem

### Reading:
- **Chapter 1 & 2** - PMPP.

### File: `addVec.cu`

## Day 2

### Summary:
Worked on a 2D matrix multiplication with custom execution configuration paramaters.
liniearize multidimentional arrays in C into 1D offset ([multidimentional array in CUDA C](https://github.com/wreckitral/100daysofgpu/blob/main/notes/dynamicArrayinCuda.md)).
each thread compute one element of the multiplication.

### Learned:
- function declarations
- kernal call
- some CUDA C syntax
- mapping 2D matrix to threads

### Reading:
- Chapter 3 - PMPP

### File: `MatMul.cu`

## Day 3

### Summary:
Implemented the Color to Grayscale Conversion for image. used opencv to handle image reading to unsigned char.
each thread compute 1 channel of one output pixel.

### Learned
- we can use C++ compiler paramaters to nvcc
- how to work with offsets
- how to work with GPU and image processing

### Reading:
- Chapter 3 - PMPP

### File: `colorToGrayscaleConversion.cu`

## Day 4

### Summary:
Implemented the blur image kernel for blurring image. used opencv to handle image reading to unsigned char and to grayscale because the kernel handle only 1 channel (grayscale).
each thread compute 1 channel of one output pixel.

### Learned
- we can use C++ compiler paramaters to nvcc
- how to work with offsets
- how to work with GPU and image processing

### Reading:
- Chapter 3 - PMPP

### File: `blur.cu`

## Day 5

### Summary:
Answered the exercise on chapter 3. analyzed the kernel difference kernel design.

### Learned:
- differentiate between 1D kernel to multidimentional kernel use case
- how to map 3D tensor to 1D array

### Reading:
- Chapter 3 exercise - PMPP

### File: `exercise-chapter-3.cu`
