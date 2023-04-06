//
// kernel.cu

///////////////////////// colToGray

__global__ void colToGray_v0(float *inImage, float *outImage, int width, int height, int numChannels) {

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

__global__ void colToGray_v1_0(float *inImage, int imageArea) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= imageArea * 3) return;
    inImage[idx] = inImage[idx] * 0.07;
}

__global__ void colToGray_v1_1(float *inImage, float *outImage, int imageArea) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= imageArea) return;

    int rgbIndex = idx * 3;

    float r = inImage[rgbIndex];
    float g = inImage[rgbIndex + 1];
    float b = inImage[rgbIndex + 2];
  
    outImage[idx] = (3 * r + 10 * g + b);
}

__global__ void colToGray_v2(float *inImage, float *outImage, int imageArea) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= imageArea) return;

    int rgbIndex = idx * 3;

    float r = inImage[rgbIndex];
    float g = inImage[rgbIndex + 1];
    float b = inImage[rgbIndex + 2];
  
    outImage[idx] = (0.21 * r + 0.71 * g + 0.07 * b);
}

  // num threads = image area * 3
__global__ void colToGray_v3_0(float *inImage, float *intImage, int imageArea) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= imageArea * 3) return;
    int outIdx = (idx % 3) * imageArea + (idx / 3);
    intImage[outIdx] = inImage[idx];
}

__global__ void colToGray_v3_1(float *intImage, float *outImage, int imageArea) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= imageArea) return;

    float r = intImage[idx];
    float g = intImage[idx + imageArea];
    float b = intImage[idx + imageArea + imageArea];
  
    outImage[idx] = (0.21 * r + 0.71 * g + 0.07 * b);
}

///////////////////////// cannyEdge

__global__ void cannyEdge_v0_0(float *imageIn, float *gradient, float *angle, int width, int height) {

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

        float grad = sqrtf( pow(gx, 2) + pow(gy, 2) );
        float theta = atan2f( gy, gx );
        theta = theta + (theta < 0) * M_PI;

        gradient[index] = grad;
        angle[index] = theta;
    }
}

__global__ void cannyEdge_v1_0(float *imageIn, float *gradient, float *angle, int width, int height) {

    // 0 1 2
    // 3   4
    // 5 6 7

    __shared__ float n[18][18];

    int col = threadIdx.x + blockIdx.x * blockDim.x; // global col
    int row = threadIdx.y + blockIdx.y * blockDim.y; // global row
    
    if (col >= width || row >= height) return;

    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 18; j++) {
            n[i][j] = 0;
        }
    }

    __syncthreads();

    int bCol = threadIdx.x; // block col
    int bRow = threadIdx.y; // block row

    int index = (row * width + col);

    n[bCol + 1][bRow + 1] = imageIn[index];

    if (bCol == 0 && col > 0)       n[0][bRow + 1] = imageIn[index - 1]; 
    if (bCol == 15 && col < width - 1)  n[17][bRow + 1] = imageIn[index + 1]; 
    if (bRow == 0 && row > 0)       n[bCol + 1][0] = imageIn[index - width];
    if (bRow == 15 && row < height - 1) n[bCol + 1][17] = imageIn[index + width];

    if (bCol == 0 && bRow == 0 && col > 0 && row > 0)            n[0][0] =  imageIn[index - width - 1];
    if (bCol == 15 && bRow == 0 && col < width - 1 && row > 0)       n[17][0] =  imageIn[index - width + 1];
    if (bCol == 0 && bRow == 15 && col > 0 && row < height - 1)      n[0][17] =  imageIn[index + width - 1];
    if (bCol == 15 && bRow == 15 && col < width - 1 && row < height - 1) n[17][17] =  imageIn[index + width + 1];

    __syncthreads();

    float gx = -n[bCol][bRow] + n[bCol+2][bRow] - (2 * n[bCol][bRow+1]) + (2 * n[bCol+2][bRow+1]) - n[bCol][bRow+2] + n[bCol+2][bRow+2];
    float gy = -n[bCol][bRow] + n[bCol][bRow+2] - (2 * n[bCol+1][bRow]) + (2 * n[bCol+1][bRow+2]) - n[bCol+2][bRow] + n[bCol+2][bRow+2];

    float grad = sqrtf( pow(gx, 2) + pow(gy, 2) );
    float theta = atan2f( gy, gx );
    theta = theta + (theta < 0) * M_PI;

    gradient[index] = grad;
    angle[index] = theta;
}

__global__ void cannyEdge_v2_0(float *imageIn, float2 *out, int width, int height) {

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

        float grad = pow(gx, 2) + pow(gy, 2);
        float theta = atan2f( gy, gx );
        theta = theta + (theta < 0) * M_PI;

        float2 val = make_float2(grad, theta);

        out[index] = val;
    }
}

__global__ void cannyEdge_v2_1(float2 *in, float *edgemap, int width, int height, float lowThresh) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int index = (row * width + col);

        float grad = in[index].x;
        float theta = in[index].y;

        float max = 1;

        if ( (theta < M_PI / 8) || (theta > (7 * M_PI / 8)) ) {
            if (col > 0 && in[index - 1].x > grad)
                max = 0;
            if (col < width - 1 && in[index + 1].x > grad)
                max = 0;
        }
        else if ( theta < (3 * M_PI / 8) ) {
            if (row > 0 && col > 0 && in[index - width - 1].x > grad)
                max = 0;
            if (row < height - 1 && col < width - 1 && in[index + width + 1].x > grad)
                max = 0;
        }
        else if ( theta < (5 * M_PI / 8) ) {
            if (row > 0 && in[index - width].x > grad)
                max = 0;
            if (row < height - 1 && in[index + width].x > grad)
                max = 0;
        }
        else {
            if (row > 0 && col < width - 1 && in[index - width + 1].x > grad)
                max = 0;
            if (row < height - 1 && col > 0 && in[index + width - 1].x > grad)
                max = 0;
        }

        if (grad < pow(lowThresh, 2)) {
            max = 0;
        }

        edgemap[index] = max;
    }
}

__global__ void cannyEdge_v0_1(float *gradient, float *angle, float *edgemap, int width, int height, float lowThresh) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < height) {
        int index = (row * width + col);

        float grad = gradient[index];
        float theta = angle[index];

        float max = 1;

        if ( (theta < M_PI / 8) || (theta > (7 * M_PI / 8)) ) {
            if (col > 0 && gradient[index - 1] > grad)
                max = 0;
            if (col < width - 1 && gradient[index + 1] > grad)
                max = 0;
        }
        else if ( theta < (3 * M_PI / 8) ) {
            if (row > 0 && col > 0 && gradient[index - width - 1] > grad)
                max = 0;
            if (row < height - 1 && col < width - 1 && gradient[index + width + 1] > grad)
                max = 0;
        }
        else if ( theta < (5 * M_PI / 8) ) {
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

///////////////////////// applyMask
__global__ void applyMask_v0(float* inEdgemap, float* outMasked, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    // Define the points of the triangle (bottom left corner, bottom right corner, center of image)
    float x1 = 0;
    float y1 = height - 1;

    float x2 = width - 1;
    float y2 = height - 1;

    float x3 = (float)width / 2;
    float y3 = (float)height / 2;

    // Compute the barycentric coordinates of the current pixel
    float alpha = ((y2 - y3)*(idx - x3) + (x3 - x2)*(idy - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
    float beta = ((y3 - y1)*(idx - x3) + (x1 - x3)*(idy - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
    float gamma = 1.0 - alpha - beta;

    // Check if the current pixel is inside the triangle
    if (alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && inEdgemap[idy * width + idx] > 0.5f) {
        outMasked[idy * width + idx] = 1.0;
    } else {
        outMasked[idy * width + idx] = 0.0;
    }
}

__global__ void applyMask_v1(float* inEdgemap, float* outMasked, int width, int height,
                               float wX, float hY, float denom) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    // Compute variables & global index
    float xA = idx - (wX + 1);
    float yA = idy - (hY + 1);
    int index = idy * width + idx;

    // Compute the barycentric coordinates of the current pixel
    float alpha = (hY * xA - wX * yA) / denom;
    float beta =  (-hY * xA - (wX+1) * yA) / denom;
    float gamma = 1.0 - alpha - beta;

    // Check if the current pixel is inside the triangle
    outMasked[index] = (alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && inEdgemap[index] > 0.5f);
}