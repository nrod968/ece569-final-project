#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define M_PI 3.14159265358979323846

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { 
			exit(code); 
			getchar(); 
		}
	}
}
__device__ inline float gaussian(float x, float sigma) {	
	return 1.0f/(sigma*sqrt(2*M_PI))*exp(-(x*x) / (2 * sigma*sigma));
}
__global__ void gpuBFCalculation(float *padimage,float *outimage,float min, float max, float *cGaussian, int pad_height, int pad_width, int kernelRadius, float sigmaD, float sigmaR) {	
	//Calculate our pixel's location
	int x=blockIdx.x*blockDim.x + threadIdx.x;	
	int y=blockIdx.y*blockDim.y + threadIdx.y;
	int out_width=pad_width - kernelRadius * 2;
	int out_height=pad_height - kernelRadius * 2;
	//Boundary check
	if (x < kernelRadius || x >= (pad_height - kernelRadius) || y < kernelRadius || y >= (pad_width - kernelRadius))
		return;

	float sum = 0;
	float totalWeight = 0;
	float centerIntensity = padimage[x*pad_width + y];
	float normCenterIntensity=(centerIntensity - min) / (max - min);
	int kernelSize=kernelRadius * 2 + 1;

	for (int dx=x - kernelRadius; dx <= x + kernelRadius; dx++) {
		for (int dy=y - kernelRadius; dy <= y + kernelRadius; dy++) {
			float kernelPosIntensity=padimage[dx*pad_width + dy];			
			float normKernelPosIntensity=(kernelPosIntensity - min) / (max - min);
			float weight= (cGaussian[dy - y + kernelRadius] * cGaussian[dx - x + kernelRadius]) * gaussian(normKernelPosIntensity - normCenterIntensity, sigmaR);				
			sum+=(weight*kernelPosIntensity);
			totalWeight+=weight;			
		}
	}	
	outimage[(x - kernelRadius) * out_width + (y - kernelRadius)] = sum / totalWeight;
}