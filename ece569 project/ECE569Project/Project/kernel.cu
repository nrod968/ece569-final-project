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

__global__ void colToGray_v2_byte(float *inImage, uint8_t *outImage, int imageArea) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= imageArea) return;

    int rgbIndex = idx * 3;

    float r = inImage[rgbIndex];
    float g = inImage[rgbIndex + 1];
    float b = inImage[rgbIndex + 2];
  
    outImage[idx] = (uint8_t)((0.21 * r + 0.71 * g + 0.07 * b) * 255);
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

__global__ void colToGray_v4(float *inImage, float *outImage, int imageArea) {
    __shared__ float s[1024];

    int i = threadIdx.x;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) - blockIdx.x;
    if (idx >= imageArea * 3) return;

    s[i] = inImage[idx];

    __syncthreads();

    if (i < 341) {
        int index = i * 3;
        outImage[i + (blockIdx.x * 341)] = (0.21 * s[index] + 0.71 * s[index+1] + 0.07 * s[index+2]);
    }
}

__host__ __device__ inline float gaussian(float x, float sigma) {	
	return 1.0f/(sigma*sqrtf(2*M_PI))*expf(-(x*x) / (2 * sigma*sigma));
}

__global__ void bilateral(float *input, float *output, float *cGaussian, int height, int width, int kernelWidth, float sigmaR) {	
	//Calculate our pixel's location
	int col=blockIdx.x*blockDim.x + threadIdx.x;	
	int row=blockIdx.y*blockDim.y + threadIdx.y;
	//Boundary check
	if (row >= height || col >= width)
		return;
	// keep a running sum of the weights * intensity of each pixel and of the weights
	float sum = 0;
	float totalWeight = 0;
	float centerIntensity = input[row * width + col]; // find the center pixel's intensity
	int kernelRadius = (int)(kernelWidth / 2);

	// iterate through the tile to get our sum and totalWeight sum
	for (int dy= -1 * kernelRadius; dy <= kernelRadius; ++dy) {
		for (int dx= -1 * kernelRadius; dx <= kernelRadius; ++dx) {
			if (row + dy < 0 || row + dy >= height || col + dx < 0 || col + dx >= width) // boundary check
				continue;
			float kernelPosIntensity=input[(row + dy)*width + (col + dx)];			
			float weight= cGaussian[(dy + kernelRadius) * kernelWidth + (dx + kernelRadius)] * gaussian(centerIntensity - kernelPosIntensity, sigmaR);				
			sum+=(weight*kernelPosIntensity);
			totalWeight+=weight;			
		}
	}	
	output[row * width + col] = sum / totalWeight;
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

__global__ void cannyEdge_v3(float *imageIn, float *edgemap, int width, int height, int lowThresh) {

    // // 0 1 2
    // // 3   4
    // // 5 6 7

    // Part 1: calculate gradient and angle

    __shared__ int ns[64][8];

    int col = (threadIdx.x + blockIdx.x * blockDim.x) - (4 * blockIdx.x) - 2;
    int row = (threadIdx.y + blockIdx.y * blockDim.y) - (4 * blockIdx.y) - 2;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int index = row * width + col;

    if (col >= 0 && row >= 0 && col < width && row < height)
        ns[tidx][tidy] = (int)(imageIn[index] * 100);
    else
        ns[tidx][tidy] = 0;

    __syncthreads();

    if (tidx >= 1 && tidy >= 1 && tidx <= blockDim.x-2 && tidy <= blockDim.y-2) {

        int gx = -ns[tidx-1][tidy-1] + ns[tidx+1][tidy-1] - (2 * ns[tidx-1][tidy]) + (2 * ns[tidx+1][tidy]) - ns[tidx-1][tidy+1] + ns[tidx+1][tidy+1];
        int gy = -ns[tidx-1][tidy-1] + ns[tidx-1][tidy+1] - (2 * ns[tidx][tidy-1]) + (2 * ns[tidx][tidy+1]) - ns[tidx+1][tidy-1] + ns[tidx+1][tidy+1];

        __syncthreads();

        int grad = (int)(sqrtf( powf(gx, 2) + powf(gy, 2) ));
        int theta = (int)(atan2f( gy, gx ) * 180 / M_PI);
        theta = theta + (theta < 0) * 180;

        ns[tidx][tidy] = grad;

        __syncthreads();

        // Part 2: find edges

        if (tidx >= 2 && tidy >= 2 && tidx <= blockDim.x-3 && tidy <= blockDim.y-3) {    
            float max = 1;

            if (grad < lowThresh) {
                max = 0;
            }
    
            if ( (theta < 22) || (theta > 157) ) {
                if (ns[tidx-1][tidy] > grad)
                    max = 0;
                if (ns[tidx+1][tidy] > grad)
                    max = 0;
            }
            else if ( theta < 67 ) {
                if (ns[tidx-1][tidy-1] > grad)
                    max = 0;
                if (ns[tidx+1][tidy+1] > grad)
                    max = 0;
            }
            else if ( theta < 112 ) {
                if (ns[tidx][tidy-1] > grad)
                    max = 0;
                if (ns[tidx][tidy+1] > grad)
                    max = 0;
            }
            else {
                if (ns[tidx+1][tidy-1] > grad)
                    max = 0;
                if (ns[tidx-1][tidy+1] > grad)
                    max = 0;
            }
    
            if (col < width && row < height)
                edgemap[index] = max;
            
        }
    }
}

__device__ float fastatan2f(float a, float b) {
    if (fabs(b - 0.0001f) > 0)
        b = 0.001f;
    double x = a / b;
    x = x * x;
    return (float)((0.077650 * x - 0.287434) * x + 0.9951816) * x;
}

__device__ float fastsinf(float a) {
    return a;
}

__device__ float fastcosf(float a) {
    return -0.4 * a * a + 1;
}

__global__ void cannyEdge_v4(float *imageIn, float *edgemap, int width, int height, int lowThresh) {
    // Part 1: calculate gradient and angle

    __shared__ uint16_t ns[16][32];
    __shared__ uint16_t grads[16][32];

    int col = (threadIdx.x + blockIdx.x * blockDim.x) - (4 * blockIdx.x) - 2;
    int row = (threadIdx.y + blockIdx.y * blockDim.y) - (4 * blockIdx.y) - 2;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int index = row * width + col;

    if (col >= 0 && row >= 0 && col < width && row < height)
        ns[tidx][tidy] = (uint16_t)(imageIn[index] * 100);
    else
        ns[tidx][tidy] = 0;

    __syncthreads();

    if (tidx >= 1 && tidy >= 1 && tidx <= blockDim.x-2 && tidy <= blockDim.y-2) {

        int gx = (int)ns[tidx+1][tidy] - (int)ns[tidx-1][tidy];
        int gy = (int)ns[tidx][tidy+1] - (int)ns[tidx][tidy-1];

        uint16_t grad = ( gx * gx + gy * gy );
        float theta = fastatan2f( gy, gx );

        grads[tidx][tidy] = grad;

        __syncthreads();

        // Part 2: find edges

        if (tidx >= 2 && tidy >= 2 && tidx <= blockDim.x-3 && tidy <= blockDim.y-3 && col < width && row < height && grad >= lowThresh) {    
            int x = (int)(fastcosf(theta) - 0.6) + (int)(fastcosf(theta) + 0.6);
            int y = (int)(fastsinf(theta) - 0.6) + (int)(fastsinf(theta) + 0.6);

            int max = fmaxf(grads[tidx + x][tidy + y], grad);
            max = fmaxf( grads[tidx - x][tidy - y], max );
            edgemap[index] = (max == grad);
        }
    }
}

__global__ void cannyEdge_v3_byte(uint8_t *imageIn, uint8_t *edgemap, int width, int height, int lowThresh) {

    // // 0 1 2
    // // 3   4
    // // 5 6 7

    // Part 1: calculate gradient and angle

    __shared__ int ns[32][8];

    int col = (threadIdx.x + blockIdx.x * blockDim.x) - (4 * blockIdx.x) - 2;
    int row = (threadIdx.y + blockIdx.y * blockDim.y) - (4 * blockIdx.y) - 2;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int index = row * width + col;

    if (col >= 0 && row >= 0 && col < width && row < height)
        ns[tidx][tidy] = (int)(imageIn[index]);
    else
        ns[tidx][tidy] = 0;

    __syncthreads();

    if (tidx >= 1 && tidy >= 1 && tidx <= blockDim.x-2 && tidy <= blockDim.y-2) {

        int gx = -ns[tidx-1][tidy-1] + ns[tidx+1][tidy-1] - (2 * ns[tidx-1][tidy]) + (2 * ns[tidx+1][tidy]) - ns[tidx-1][tidy+1] + ns[tidx+1][tidy+1];
        int gy = -ns[tidx-1][tidy-1] + ns[tidx-1][tidy+1] - (2 * ns[tidx][tidy-1]) + (2 * ns[tidx][tidy+1]) - ns[tidx+1][tidy-1] + ns[tidx+1][tidy+1];

        int grad = (int)(sqrtf( powf(gx, 2) + powf(gy, 2) ));
        int theta = (int)(atan2f( gy, gx ) * 180 / M_PI);
        theta = theta + (theta < 0) * 180;

        __syncthreads();

        ns[tidx][tidy] = grad;

        __syncthreads();

        // Part 2: find edges
    
        if (tidx >= 2 && tidy >= 2 && tidx <= blockDim.x-3 && tidy <= blockDim.y-3) {    
            uint8_t max = 1;
    
            if ( (theta < 22) || (theta > 157) ) {
                if (ns[tidx-1][tidy] > grad)
                    max = 0;
                if (ns[tidx+1][tidy] > grad)
                    max = 0;
            }
            else if ( theta < 67 ) {
                if (ns[tidx-1][tidy-1] > grad)
                    max = 0;
                if (ns[tidx+1][tidy+1] > grad)
                    max = 0;
            }
            else if ( theta < 112 ) {
                if (ns[tidx][tidy-1] > grad)
                    max = 0;
                if (ns[tidx][tidy+1] > grad)
                    max = 0;
            }
            else {
                if (ns[tidx+1][tidy-1] > grad)
                    max = 0;
                if (ns[tidx-1][tidy+1] > grad)
                    max = 0;
            }
    
            if (grad < lowThresh) {
                max = 0;
            }
    
            if (col < width && row < height)
                edgemap[index] = max;
            
        }
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

__global__ void applyMask_v1_byte(uint8_t* inEdgemap, float* outMasked, int width, int height,
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
outMasked[index] = (alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && inEdgemap[index] == 1);
}

//Filter lines into positive and negative slope
__global__ void filterLines(float* lines, int numLines, int maxLines, float* posLines, float* negLines){
    bool negAdded = false;
    bool posAdded = true;

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    //if thread is not in range of max lines then don't perform calculations
    if(tid >= numLines){
        return;
    }
        //get x1,y1,x2,y2,slope for line
        float x1 = lines[tid];
        float y1 = lines[tid + maxLines];
        float x2 = lines[tid + (maxLines * 2)];
        float y2 = lines[tid + (maxLines * 3)];
        float slope = lines[tid + (maxLines * 4)];

        float xdiff = x2 - x1;
        float ydiff = y2 - y1;

        //calculate length of line using sqrt((x2 - x1)^2 + (y2 - y1)^2)
        float lineLength = hypotf(xdiff, ydiff);

        //if line is long enough
        if(lineLength > 30){
            if(x1 != x2){

                //if positive sloped
                if(slope > 0){
                    float tanTheta = tanf(fabsf(ydiff)/fabsf(xdiff));
                    float angle = atanf(tanTheta) * (180/3.14);

                    //write line to posLines array
                    if(fabsf(angle) > 20 && fabsf(angle) < 85){
                        posLines[tid] = x1;
                        posLines[tid + maxLines] = y1;
                        posLines[tid + (maxLines * 2)] = x2;
                        posLines[tid + (maxLines * 3)] = y2;
                        posLines[tid + (maxLines * 4)] = slope;
                        posAdded = true;
                    }
                }

                //if negative sloped
                if(slope < 0){
                    float tanTheta = tanf(fabsf(ydiff)/fabsf(xdiff));
                    float angle = atanf(tanTheta) * (180/3.14);

                    //write line to negLines array
                    if(fabsf(angle) > 20 && fabsf(angle) < 85){
                        negLines[tid] = x1;
                        negLines[tid + maxLines] = y1;
                        negLines[tid + (maxLines * 2)] = x2;
                        negLines[tid + (maxLines * 3)] = y2;
                        negLines[tid + (maxLines * 4)] = slope;
                        negAdded = true;
                    }
                }
            }
        }

        //if no positive line added then perform above calculations again for line of any length
        if(!posAdded){
            if(slope > 0){
                    float tanTheta = tanf(fabsf(ydiff)/fabsf(xdiff));
                    float angle = atanf(tanTheta) * (180/3.14);

                    //write line to posLines array
                    if(fabsf(angle) > 20 && fabsf(angle) < 85){
                        posLines[tid] = x1;
                        posLines[tid + maxLines] = y1;
                        posLines[tid + (maxLines * 2)] = x2;
                        posLines[tid + (maxLines * 3)] = y2;
                        posLines[tid + (maxLines * 4)] = slope;
                        posAdded = true;
                    }
                }
            }

        if(!negAdded){
            if(slope < 0){
                    float tanTheta = tanf(fabsf(ydiff)/fabsf(xdiff));
                    float angle = atanf(tanTheta) * (180/3.14);

                    //write line to negLines array
                    if(fabsf(angle) > 20 && fabsf(angle) < 85){
                        negLines[tid] = x1;
                        negLines[tid + maxLines] = y1;
                        negLines[tid + (maxLines * 2)] = x2;
                        negLines[tid + (maxLines * 3)] = y2;
                        negLines[tid + (maxLines * 4)] = slope;
                        negAdded = true;
                    }
                }
            }
    }


//kernel to find pos/neg slope mean from only good lines
__global__ void findSlopeMean(float* posLines, float* negLines, float* posSum, int numGoodPosLines, float* posSlopeMean, float* negSum, int numGoodNegLines, float* negSlopeMean, int maxLines){
    float posMedian;
    int posMedianIndex;
    int posLineSize = sizeof(posLines);
    float negMedian;
    int negMedianIndex;
    int negLineSize = sizeof(negLines);

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    //arrays of positive and negative lines are sorted in host code

    //only need 1 thread to calculate positive median
    if(tid == 0){
        //if even number of positive lines
        if(posLineSize % 2 == 0){
            posMedianIndex = posLineSize/2;
            //take average of middle 2 values for median
            posMedian = (posLines[posMedianIndex] + posLines[posMedianIndex - 1])/2;
        }

        else{
            posMedianIndex = posLineSize/2;
            posMedian = (posLines[posMedianIndex];
        }
    }

        //only need 1 thread to calculate positive median
    if(tid == 1){
        //if even number of negative lines
        if(negLineSize % 2 == 0){
            negMedianIndex = negLineSize/2;
            //take average of middle 2 values for median
            negMedian = (negLines[negMedianIndex] + negLines[negMedianIndex - 1])/2;
        }

        else{
            negMedianIndex = negLineSize/2;
            negMedian = (negLines[negMedianIndex];
        }
    }

    //for threads in pos lines size only
    if(tid <= posLineSize){
        if(fabsf(posLines[tid] - posMedian) < (posMedian * 0.2)){
            atomicAdd(&numPosGoodLines, 1);
            atomicAdd(&posSum, posLines[tid]);
        }
    }

    //for threads in neg line size only
    if(tid >= posLineSize && tid <= maxLines){
        if(fabsf(negLines[tid] - negMedian) < (negMedian * 0.9)){
            atomicAdd(&numNegGoodLines, 1);
            atomicAdd(&negSum, negLines[tid]);
        }
        __syncthreads();

        negSlopeMean = negSum/numNegGoodLines;
    }

    __syncthreads();

    if(tid == 0){
        posSlopeMean = posSum/numPosGoodLines;
    }

    if(tid == 1){
        negSlopeMean = negSum/numNegGoodLines;
    }
}

//finds x intercepts for all positive and negative lines
__global__ void findXIntercepts(float* posLines, float* negLines, float* xInterceptPos, float* xInterceptNeg, int numPosLines, int numNegLines int maxLines){
    float posxIntercept;
    float posyIntercept;
    float negxIntercept;
    float negyIntercept;

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid < numPosLines){
        float posx1 = posLines[tid];
        float posy1 = posLines[tid + maxLines];
        float posSlope = posLines[tid + (maxLines * 4)];
        posyIntercept = posy1 - posSlope * posx1;
        posxIntercept = -yIntercept/posSlope;

        xInterceptPos[tid] = posxIntercept;
    }

    if(tid >= numPosLines && tid < maxLines){
        float negx1 = negLines[tid];
        float negy1 = negLines[tid + maxLines];
        float negSlope = negLines[tid + (maxLines * 4)];
        negyIntercept = negy1 - negSlope * negx1;
        negxIntercept = -negyIntercept/negSlope;

        xInterceptNeg[tid] = negxIntercept;
    }
}

//finds mean at x intercepts using only good lines
//array sorted and median calculated outside kernel
__global__ void findMeanXatY0(float xIntPosMedian, float xIntNegMedian, float xIntPosMean, float xIntNegMean, 
    int numPosLines, int numNegLines, float* xInterceptPos, float* xInterceptNeg, 
    int numPosGood, int numNegGood, float posxIntSum, float negxIntSum int maxLines){

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid < numPosLines){
        float posxIntercept = xInterceptPos[tid];
        if(fabsf(posxIntercept - xIntPosMedian) < (0.35 * xIntPosMedian)){
            xIntPosGood[tid] = posxIntercept;
            atomicAdd(&posxIntSum, posxIntercept);
            atomicAdd(&numPosGood, 1);
        }
    }

    if(tid >= numPosLines && tid < maxLines){
        float negxIntercept = xInterceptNeg[tid];
        if(fabsf(negxIntercept - xIntNegMedian) < (0.35 * xIntNegMedian)){
            xIntPosGood[tid] = posxIntercept;
            atomicAdd(&negxIntSum, negxIntercept);
            atomicAdd(&numNegGood, 1);
        }
    }
    __syncthreads();

    if(tid == 0){
        xIntPosMean = posxIntSum/numPosGood;
    }

    if(tid == 1){
        xIntNegMean = negxIntSum/numNegGood;
    }
}

__global__ void plotLaneLines(float* image, float posSlopeMean, float xIntPosMean, float negSlopeMean, float xIntNegMean, 
    int imgHeight){

    //Positive coordinates
    float x1Pos = xIntPosMean;
    float y1Pos = 0;
    float y2Pos = imgHeight - (imgHeight - (imgHeight *0.35));
    float x2Pos = (y2Pos - y1Pos)/posSlopeMean + x1Pos;

    //Negative coordinates
    float x1Neg = xIntNegMean;
    float y1Neg = 0;
    float y2Neg = imgHeight - (imgHeight - (imgHeight *0.35));
    float x2Neg = (y2Neg - y1)Neg/negSlopeMean + x1Neg;

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    x1Pos = int(x1Pos + .5);
	x2Pos = int(x2Pos + .5);
	y1Pos = int(y1Pos + .5);
	y2Pos = int(y2Pos + .5);

    //color positive line
    if(tid >= x1Pos && tid <= x2Pos && tid >= y1Pos && tid <= y2Pos){
        image[tid] = 0.23;
    }

    x1Neg = int(x1Neg + .5);
	x2Neg = int(x2Neg + .5);
	y1Neg = int(y1Neg + .5);
	y2Neg = int(y2Neg + .5);

    //color negative line
    if(tid >= x1Neg && tid <= x2Neg && tid >= y1Neg && tid <= y2Neg){
        image[tid] = 0.23;
    }
}

