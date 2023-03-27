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

        float grad = sqrtf( pow(gx, 2) + pow(gy, 2) );
        float theta = atan2f( gy, gx );
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

__global__ void apply_mask(float* inEdgemap, float* outMasked, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    // Define the points of the triangle
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