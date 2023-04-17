#include <math.h>
#define M_PI 3.14159265358979323846

__host__ __device__ inline float gaussian(float x, float sigma) {	
	return 1.0f/(sigma*sqrtf(2*M_PI))*expf(-(x*x) / (2 * sigma*sigma));
}

__global__ void gpuBFCalculation(float *input,float *output, float *cGaussian, int height, int width, int kernelWidth, float sigmaR) {	
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

float* BFLaunch(float* input, int width, int height, float sigmaS, float sigmaR){
	int kernelRadius = log2f(min(width, height));
	int kernelWidth = kernelRadius * 2 + 1;
	float* fGaussian = (float*)malloc(kernelWidth * kernelWidth * sizeof(float));
	float *d_cGaussian;
	float* output,
	for (int i = 0; i < kernelWidth; ++i){
		for (int j = 0; j < kernelWidth; ++i){
			float x = sqrtf((i - kernelRadius) * (i - kernelRadius) + (j - kernelRadius) * (j - kernelRadius));
			fGaussian[i * kernelWidth + j] = gaussian(x, sigmaS);
		}
	}
	cudaMalloc(&d_cGaussian, sizeof(float)*(kernelWidth * kernelWidth));
	cudaMemcpy(d_cGaussian, fGaussian, sizeof(float)*(kernelWidth * kernelWidth), cudaMemcpyHostToDevice);
	free(fGaussian);

	float *d_input;
	float *d_output;
	//Cuda memory allocation and error check
	gpuErrchk(cudaMalloc(&d_input, sizeof(float)*width*height));//GPU-memory allocation for d_padimage
	gpuErrchk(cudaMemcpy(d_input, input, sizeof(float)*width*height, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_output, sizeof(float)*width*height));
	dim3 threadsPerBlock(16,16);//normally 16*16 is optimal
	dim3 numBlocks(ceil((float)height / threadsPerBlock.x), ceil((float)width / threadsPerBlock.y)); 
	gpuBFCalculation <<<numBlocks, threadsPerBlock >>> (d_input, d_output, d_cGaussian, height, width, kernelWidth, sigmaR);
	gpuErrchk(cudaMemcpy(output, d_output, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_cGaussian);

	return outimage;
}