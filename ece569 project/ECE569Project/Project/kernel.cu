#define PI 3.141592654f

__global__ void colToGray(float *inImage, float *outImage, int width, int height, int numChannels) {

    // Determine column and row of thread
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    // If within bounds of image, perform grayscale operation.
    if (col < width && row < height) {
      int index = (row * width + col);
      int rgbIndex = index * numChannels;
  
      float r = inImage[rgbIndex];
      float g = inImage[rgbIndex + 1];
      float b = inImage[rgbIndex + 2];
  
      outImage[index] = (0.21 * r + 0.71 * g + 0.07 * b);
    }
  }

__global__ void canny_edge_0(float *imageIn, float *gradient, float *angle, int width, int height) {

    // 0 1 2
    // 3   4
    // 5 6 7

    float n[8];

    for (int i = 0; i < 8; i++) {
        n[i] = 0;
    }

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int index = (row * width + col);

        if (col > 0) {
            if (row > 0)
                n[0] = imageIn[index - width - 1];

            if (row < height - 1)
                n[5] = imageIn[index + width - 1]; 

            n[3] = imageIn[index - 1];
        }

        if (col < width - 1) {
            if (row > 0)
                n[2] = imageIn[index - width + 1];

            if (row < height - 1)
                n[7] = imageIn[index + width + 1];

            n[4] = imageIn[index + 1];
        }

        if (row > 0) 
            n[1] = imageIn[index - width];

        if (row < height - 1)
            n[6] = imageIn[index + width];

        float gx = -n[0] + n[2] - (2 * n[3]) + (2 * n[4]) - n[5] + n[7];
        float gy = -n[0] + n[5] - (2 * n[1]) + (2 * n[6]) - n[2] + n[7];

        float grad = sqrt( pow(gx, 2) + pow(gy, 2) );
        float theta = atan2( gy, gx );
        theta = theta + (theta < 0) * PI;

        gradient[index] = grad;
        angle[index] = theta;
    }
}

__global__ void canny_edge_1(float *gradient, float *angle, float *edgemap, int width, int height, float lowThresh) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int index = (row * width + col);

        float grad = gradient[index];
        float theta = angle[index];

        float max = 1;

        if ( (theta < PI / 8) || (theta > (7 * PI / 8)) ) {
            if (col > 0 && gradient[index - 1] > grad)
                max = 0;
            if (col < width - 1 && gradient[index + 1] > grad)
                max = 0;
        }
        else if ( theta < (3 * PI / 8) ) {
            if (row > 0 && col > 0 && gradient[index - width - 1] > grad)
                max = 0;
            if (row < height - 1 && col < width - 1 && gradient[index + width + 1] > grad)
                max = 0;
        }
        else if ( theta < (5 * PI / 8) ) {
            if (row > 0 && gradient[index - width] > grad)
                max = 0;
            if (row < height - 1 && gradient[index + width] > grad)
                max = 0;
        }
        else {
            if (row > 0 && col < width - 1 && gradient[index - width + 1] > grad)
                max = 0;
            if (row < height - 1 && col > 0 && gradient[index + width - 1] > grad)
                max = 0;
        }

        if (grad < lowThresh) {
            max = 0;
        }

        edgemap[index] = max;
    }
}



// __device__ void gaussian_kernel(float* kernel, int size, float sigma) {
//     float sum = 0.0f;
//     int offset = size / 2;

//     for (int i = -offset; i <= offset; i++) {
//         for (int j = -offset; j <= offset; j++) {
//             float r = sqrtf(i * i + j * j);
//             float val = expf(-r * r / (2.0f * sigma * sigma)) / (2.0f * M_PI * sigma * sigma);
//             kernel[(i + offset) * size + j + offset] = val;
//             sum += val;
//         }
//     }

//     for (int i = 0; i < size * size; i++) {
//         kernel[i] /= sum;
//     }
// }

// __device__ void apply_kernel(float* output, const float* input, const float* kernel, int size, int width, int height) {
//     int offset = size / 2;

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x;
//     int by = blockIdx.y * blockDim.y;
//     int x = bx + tx;
//     int y = by + ty;

//     float sum = 0.0f;

//     if (x < width && y < height) {
//         for (int i = -offset; i <= offset; i++) {
//             for (int j = -offset; j <= offset; j++) {
//                 int idx = (y + i) * width + x + j;
//                 if (idx >= 0 && idx < width * height) {
//                     float val = input[idx];
//                     float weight = kernel[(i + offset) * size + j + offset];
//                     sum += val * weight;
//                 }
//             }
//         }
//         output[y * width + x] = sum;
//     }
// }

// __device__ void sobel_filter(float* grad_x, float* grad_y, const float* input, int width, int height) {
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x;
//     int by = blockIdx.y * blockDim.y;
//     int x = bx + tx;
//     int y = by + ty;

//     float dx = 0.0f;
//     float dy = 0.0f;

//     if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
//         dx += input[(y - 1) * width + (x + 1)] - input[(y - 1) * width + (x - 1)] +
//               2.0f * input[y * width + (x + 1)] - 2.0f * input[y * width + (x - 1)] +
//               input[(y + 1) * width + (x + 1)] - input[(y + 1) * width + (x - 1)];

//         dy += input[(y - 1) * width + (x - 1)] + 2.0f * input[(y - 1) * width + x] + input[(y - 1) * width + (x + 1)] -
//               input[(y + 1) * width + (x - 1)] - 2.0f * input[(y + 1) * width + x] - input[(y + 1) * width + (x + 1)];

//         grad_x[y * width + x]

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <cuda_runtime.h>

// __device__ void gaussian_kernel(float* kernel, int size, float sigma) {
//     float sum = 0.0f;
//     int offset = size / 2;

//     for (int i = -offset; i <= offset; i++) {
//         for (int j = -offset; j <= offset; j++) {
//             float r = sqrtf(i * i + j * j);
//             float val = expf(-r * r / (2.0f * sigma * sigma)) / (2.0f * M_PI * sigma * sigma);
//             kernel[(i + offset) * size + j + offset] = val;
//             sum += val;
//         }
//     }

//     for (int i = 0; i < size * size; i++) {
//         kernel[i] /= sum;
//     }
// }

// __device__ void apply_kernel(float* output, const float* input, const float* kernel, int size, int width, int height) {
//     int offset = size / 2;

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x;
//     int by = blockIdx.y * blockDim.y;
//     int x = bx + tx;
//     int y = by + ty;

//     float sum = 0.0f;

//     if (x < width && y < height) {
//         for (int i = -offset; i <= offset; i++) {
//             for (int j = -offset; j <= offset; j++) {
//                 int idx = (y + i) * width + x + j;
//                 if (idx >= 0 && idx < width * height) {
//                     float val = input[idx];
//                     float weight = kernel[(i + offset) * size + j + offset];
//                     sum += val * weight;
//                 }
//             }
//         }
//         output[y * width + x] = sum;
//     }
// }

// __device__ void sobel_filter(float* grad_x, float* grad_y, const float* input, int width, int height) {
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x;
//     int by = blockIdx.y * blockDim.y;
//     int x = bx + tx;
//     int y = by + ty;

//     float dx = 0.0f;
//     float dy = 0.0f;

//     if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
//         dx += input[(y - 1) * width + (x + 1)] - input[(y - 1) * width + (x - 1)] +
//               2.0f * input[y * width + (x + 1)] - 2.0f * input[y * width + (x - 1)] +
//               input[(y + 1) * width + (x + 1)] - input[(y + 1) * width + (x - 1)];

//         dy += input[(y - 1) * width + (x - 1)] + 2.0f * input[(y - 1) * width + x] + input[(y - 1) * width + (x + 1)] -
//               input[(y + 1) * width + (x - 1)] - 2.0f * input
