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

  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);

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

  // Define threads per block to be 16 x 16, and number of blocks to accomodate entire image.
  dim3 g16x16( ceil(imageWidth / 16.0), ceil(imageHeight / 16.0) );
  dim3 b16x16(16, 16);
  dim3 g128( ceil(imageWidth * imageHeight / 128.0));
  dim3 b128(128);
  dim3 g128_tri( ceil(imageWidth * imageHeight * 3 / 128.0));
  dim3 g64x8( ceil(imageWidth / 64.0), ceil(imageHeight / 8.0) );
  dim3 b64x8(64, 8);

  /////////////////////////////////////////////////////// COL TO GRAY

  // colToGray_v0
  {
    cudaEventRecord(astartEvent, 0);

    colToGray_v0<<<g16x16, b16x16>>>(deviceInputImageData, gsData, imageWidth, imageHeight, imageChannels);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v0 \n", aelapsedTime);
  }

  // colToGray_v1
  {
    cudaEventRecord(astartEvent, 0);

    colToGray_v1<<<g128_tri, b128>>>(deviceInputImageData, imageWidth * imageHeight);
    colToGray_v1<<<g128, b128>>>(deviceInputImageData, gsData, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v1 \n", aelapsedTime);
  }
  
  // colToGray_v2
  {
    cudaEventRecord(astartEvent, 0);

    colToGray_v2<<<g128, b128>>>(deviceInputImageData, gsData, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v2 \n", aelapsedTime);
  }
  
  // colToGray_v3
  {
    float *intImageData;
    cudaMalloc((void **)&intImageData, imageWidth * imageHeight * 3 * sizeof(float));
    
    cudaEventRecord(astartEvent, 0);

    colToGray_v3<<<g128_tri, b128>>>(deviceInputImageData, intImageData, imageWidth * imageHeight);
    colToGray_v3<<<g128, b128>>>(intImageData, gsData, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v3 \n", aelapsedTime);
    cudaFree(intImageData);
  }

  /////////////////////////////////////////////////////// CANNY

  float *gradData;
  float *angleData;
  float *edgeData;

  cudaMalloc((void **)&gradData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&angleData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&edgeData, imageWidth * imageHeight * sizeof(float));

  // cannyEdge_v0
  {
    cudaEventRecord(astartEvent, 0);

    cannyEdge_v0_0<<<g16x16, b16x16>>>(gsData, gradData, angleData, imageWidth, imageHeight);
    cannyEdge_v0_1<<<g16x16, b16x16>>>(gradData, angleData, edgeData, imageWidth, imageHeight, 0.2353f);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v0 \n", aelapsedTime);
  }

  // cannyEdge_v0 (Long)
  {
    cudaEventRecord(astartEvent, 0);

    canny_edge_0<<<g64x8, b64x8>>>(gsData, gradData, angleData, imageWidth, imageHeight);
    canny_edge_1<<<g64x8, b64x8>>>(gradData, angleData, edgeData, imageWidth, imageHeight, 0.2353f);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v0 (Long) \n", aelapsedTime);
  }

  // cannyEdge_v1
  {
    cudaEventRecord(astartEvent, 0);

    canny_edge_v1_0<<<g16x16, b16x16>>>(gsData, gradData, angleData, imageWidth, imageHeight);
    canny_edge_v0_1<<<g16x16, b16x16>>>(gradData, angleData, edgeData, imageWidth, imageHeight, 0.2353f);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v1 \n", aelapsedTime);
  }
  
  // cannyEdge_v2
  {
    float2 *gradangleData;
    cudaMalloc((void **)&gradangleData, imageWidth * imageHeight * sizeof(float2));

    cudaEventRecord(astartEvent, 0);

    cannyEdge_v2_0<<<g64x8, b64x8>>>(gsData, gradangleData, imageWidth, imageHeight);
    cannyEdge_v2_1<<<g64x8, b64x8>>>(gradangleData, edgeData, imageWidth, imageHeight, 0.2353f);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v2 \n", aelapsedTime);
    cudaFree(gradangleData);
  }

  cudaFree(gsData);
  cudaFree(gradData);
  cudaFree(angleData);
 

  /////////////////////////////////////////////////////// MASKING

  float *maskData;

  cudaMalloc((void **)&maskData, imageWidth * imageHeight * sizeof(float));
  
  // applyMask_v0
  {
    cudaEventRecord(astartEvent, 0);

    applyMask_v0<<<g16x16, b16x16>>>(edgeData, maskData, imageWidth, imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for applyMask_v0 \n", aelapsedTime);
  }

  // applyMask_v1
  {
    float wX = ((float)imageWidth / 2) - 1;                  // (width - 1) - (width / 2)
    float hY = ((float)imageHeight / 2);                     // (height - 1) - (height / 2) + 1
    float denom = (hY + imageWidth) - (hY * imageWidth) - 1; // Calculates barycentric denominator
    hY -= 1;                                                 // (height - 1) - (height / 2)

    cudaEventRecord(astartEvent, 0);

    apply_mask_opt<<<g16x16, b16x16>>>(edgeData, maskData, imageWidth, imageHeight, wX, hY, denom);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for applyMask_v1 \n", aelapsedTime);
  }

  cudaFree(edgeData);


  /////////////////////////////////////////////////////// END

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
