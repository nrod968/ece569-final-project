#define M_PI 3.14159265358979323846

__device__ inline float gaussian(float x, float sigma) {	
	return 1.0f/(sigma*sqrtf(2*M_PI))*expf(-(x*x) / (2 * sigma*sigma));
}

__global__ void gpuBFCalculation(float *input,float *output, float *cGaussian, int height, int width, int kernelWidth, float sigmaR) {	
	//Calculate our pixel's location
	int col=blockIdx.x*blockDim.x + threadIdx.x;	
	int row=blockIdx.y*blockDim.y + threadIdx.y;
	//Boundary check
	if (row >= height || col >= width)
		return;

	float sum = 0;
	float totalWeight = 0;
	float centerIntensity = input[row * width + col];

	for (int dy= -1 * (kernelWidth / 2); dy <= kernelWidth / 2; dy++) {
		for (int dx= -1 * (kernelWidth / 2); dx <= kernelWidth / 2; dx++) {
			if (row + dy < 0 || row + dy >= height || col + dx < 0 || col + dx >= width)
				continue;
			float kernelPosIntensity=input[(row + dy)*width + (col + dx)];			
			float weight= cGaussian[(dy + kernelWidth / 2) * kernelWidth + (dx + kernelWidth / 2)] * gaussian(kernelPosIntensity - centerIntensity, sigmaR);				
			sum+=(weight*kernelPosIntensity);
			totalWeight+=weight;			
		}
	}	
	outimage[row * width + col] = sum / totalWeight;
}