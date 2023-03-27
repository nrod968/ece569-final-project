//
// Noah, Yousuf, Nathaniel
// ECE 569 Project
// 
// 

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <wb.h>
#include <png.h>
#include "kernel.cu"
#include "util.cu"

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;

  // PHASE 0
  // Loading the ppm image

  int imageChannels;
  int imageWidth;
  int imageHeight;

  char *inputImageFile;

  wbImage_t inputImage;
  wbImage_t outputImage;

  float *hostInputImageData;
  float *hostOutputImageData;
  
  float *deviceInputImageData;
  //float *deviceOutputImageData;
  float *gsData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));

  cudaMalloc((void **)&gsData,
             imageWidth * imageHeight * sizeof(float));

  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);

  ///////////////////////////////////////////////////////
 
  // Define threads per block to be 16 x 16, and number of blocks to accomodate entire image.
  dim3 numBlocks( ceil(imageWidth / 16.0), ceil(imageHeight / 16.0) );
  dim3 tpb(16, 16);

  // Launch colToGray
  colToGray<<<numBlocks, tpb>>>(deviceInputImageData, gsData, imageWidth, imageHeight, imageChannels);

  ///////////////////////////////////////////////////////

    // CANNY

    float *gradData;
    float *angleData;
    float *edgeData;

    cudaMalloc((void **)&gradData, imageWidth * imageHeight * sizeof(float));
    cudaMalloc((void **)&angleData, imageWidth * imageHeight * sizeof(float));
    cudaMalloc((void **)&edgeData, imageWidth * imageHeight * sizeof(float));

  canny_edge_0<<<numBlocks, tpb>>>(gsData, gradData, angleData, imageWidth, imageHeight);
  canny_edge_1<<<numBlocks, tpb>>>(gradData, angleData, edgeData, imageWidth, imageHeight, 0.2353f);

  cudaFree(gsData);
  cudaFree(gradData);
  cudaFree(angleData);

  float *maskData;
  cudaMalloc((void **)&maskData, imageWidth * imageHeight * sizeof(float));

  apply_mask<<<numBlocks, tpb>>>(edgeData, maskData, imageWidth, imageHeight);

  cudaFree(edgeData);

  cudaMemcpy(hostOutputImageData, maskData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);

  save_image_to_pgm("outc.pbm", outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(maskData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
