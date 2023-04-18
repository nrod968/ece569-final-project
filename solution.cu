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
  uint8_t *gsDataByte;

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

  cudaMalloc((void **)&gsDataByte,
             imageWidth * imageHeight * sizeof(uint8_t));

  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);

  // Define threads per block to be 16 x 16, and number of blocks to accomodate entire image.
  dim3 g16x16( ceil(imageWidth / 16.0), ceil(imageHeight / 16.0) );
  dim3 b16x16(16, 16);
  dim3 g128( ceil(imageWidth * imageHeight / 128.0));
  dim3 b128(128);
  dim3 g64_tri( ceil(imageWidth * imageHeight * 3 / 64.0));
  dim3 b64(64);
  dim3 g256_tri( ceil(imageWidth * imageHeight * 3 / 256.0));
  dim3 b256(256);
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

  // // colToGray_v1 - Messes up the rest of the code.
  // {
  //   cudaEventRecord(astartEvent, 0);

  //   colToGray_v1_0<<<g128_tri, b128>>>(deviceInputImageData, imageWidth * imageHeight);
  //   colToGray_v1_1<<<g128, b128>>>(deviceInputImageData, gsData, imageWidth * imageHeight);

  //   cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  //   printf("\nTotal compute time (ms) %f for colToGray_v1 \n", aelapsedTime);
  // }
  
  // colToGray_v2
  {
    cudaEventRecord(astartEvent, 0);

    colToGray_v2<<<g128, b128>>>(deviceInputImageData, gsData, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v2 \n", aelapsedTime);
  }

  // colToGray_v2_byte
  {
    cudaEventRecord(astartEvent, 0);

    colToGray_v2_byte<<<g128, b128>>>(deviceInputImageData, gsDataByte, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v2_byte \n", aelapsedTime);
  }
  
  // colToGray_v3
  {
    float *intImageData;
    cudaMalloc((void **)&intImageData, imageWidth * imageHeight * 3 * sizeof(float));
    
    cudaEventRecord(astartEvent, 0);

    colToGray_v3_0<<<g128_tri, b128>>>(deviceInputImageData, intImageData, imageWidth * imageHeight);
    colToGray_v3_1<<<g128, b128>>>(intImageData, gsData, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v3 \n", aelapsedTime);
    cudaFree(intImageData);
  }

  // colToGray_v4
  {
    cudaEventRecord(astartEvent, 0);

    colToGray_v4<<<ceil(imageWidth * imageHeight * 3 / 1024.0), 1024>>>(deviceInputImageData, gsData, imageWidth * imageHeight);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for colToGray_v4 \n", aelapsedTime);
  }

  /////////////////////////////////////////////////////// CANNY

  float *gradData;
  float *angleData;
  float *edgeData;
  uint8_t *edgeDataByte;

  cudaMalloc((void **)&gradData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&angleData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&edgeData, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&edgeDataByte, imageWidth * imageHeight * sizeof(uint8_t));

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

    cannyEdge_v0_0<<<g64x8, b64x8>>>(gsData, gradData, angleData, imageWidth, imageHeight);
    cannyEdge_v0_1<<<g64x8, b64x8>>>(gradData, angleData, edgeData, imageWidth, imageHeight, 0.2353f);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v0 (Long) \n", aelapsedTime);
  }

  // cannyEdge_v1
  {
    cudaEventRecord(astartEvent, 0);

    cannyEdge_v1_0<<<g16x16, b16x16>>>(gsData, gradData, angleData, imageWidth, imageHeight);
    cannyEdge_v0_1<<<g16x16, b16x16>>>(gradData, angleData, edgeData, imageWidth, imageHeight, 0.2353f);

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

  // cannyEdge_v3
  {
    cudaEventRecord(astartEvent, 0);

    dim3 ce3g( ceil(imageWidth / 60.0), ceil(imageHeight / 4.0) );
    dim3 ce3b(64, 8);

    cannyEdge_v3<<<ce3g, ce3b>>>(gsData, edgeData, imageWidth, imageHeight, 23);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v3 \n", aelapsedTime);
  }

   // cannyEdge_v4
   {
    cudaEventRecord(astartEvent, 0);

    dim3 ce3g( ceil(imageWidth / 12.0), ceil(imageHeight / 28.0) );
    dim3 ce3b(16, 32);

    cannyEdge_v4<<<ce3g, ce3b>>>(gsData, edgeData, imageWidth, imageHeight, 529);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v4 \n", aelapsedTime);
  }

  // cannyEdge_v3_byte
  {
    cudaEventRecord(astartEvent, 0);

    dim3 ce3g( ceil(imageWidth / 28.0), ceil(imageHeight / 4.0) );
    dim3 ce3b(32, 8);

    cannyEdge_v3_byte<<<ce3g, ce3b>>>(gsDataByte, edgeDataByte, imageWidth, imageHeight, 60);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for cannyEdge_v3_byte \n", aelapsedTime);
  }

  cudaFree(gsData);
  cudaFree(gsDataByte);
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

    applyMask_v1<<<g16x16, b16x16>>>(edgeData, maskData, imageWidth, imageHeight, wX, hY, denom);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for applyMask_v1 \n", aelapsedTime);
  }

  // // applyMask_v1_byte
  // {
  //   float wX = ((float)imageWidth / 2) - 1;                  // (width - 1) - (width / 2)
  //   float hY = ((float)imageHeight / 2);                     // (height - 1) - (height / 2) + 1
  //   float denom = (hY + imageWidth) - (hY * imageWidth) - 1; // Calculates barycentric denominator
  //   hY -= 1;                                                 // (height - 1) - (height / 2)

  //   cudaEventRecord(astartEvent, 0);

  //   applyMask_v1_byte<<<g16x16, b16x16>>>(edgeDataByte, maskData, imageWidth, imageHeight, wX, hY, denom);

  //   cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  //   printf("\nTotal compute time (ms) %f for applyMask_v1_byte \n", aelapsedTime);
  // }

  cudaFree(edgeData);
  cudaFree(edgeDataByte);

  /////////////////////////////////////////////////////// HOUGH

  int ncols = findNCols(imageWidth, imageHeight);

  int *hArray;
  int *xMins;
  int *yMins;
  int *xMaxs;
  int *yMaxs;

  float *linesHost = (float*)malloc(MAX_LINES * 5 * sizeof(float));
  int numLines;

  int *hArrayHost = (int*)malloc(N_ROWS * ncols * sizeof(int));
  int *xMinsHost = (int*)malloc(N_ROWS * ncols * sizeof(int));
  int *yMinsHost = (int*)malloc(N_ROWS * ncols * sizeof(int));
  int *xMaxsHost = (int*)malloc(N_ROWS * ncols * sizeof(int));
  int *yMaxsHost = (int*)malloc(N_ROWS * ncols * sizeof(int));

  cudaMalloc((void **)&hArray, N_ROWS * ncols * sizeof(int));
  cudaMalloc((void **)&xMins,  N_ROWS * ncols * sizeof(int));
  cudaMalloc((void **)&yMins,  N_ROWS * ncols * sizeof(int));
  cudaMalloc((void **)&xMaxs,  N_ROWS * ncols * sizeof(int));
  cudaMalloc((void **)&yMaxs,  N_ROWS * ncols * sizeof(int));

  cudaMemset((void **)&hArray, 0, N_ROWS * ncols * sizeof(int));
  cudaMemset((void **)&xMins,  255, N_ROWS * ncols * sizeof(int));
  cudaMemset((void **)&yMins,  255, N_ROWS * ncols * sizeof(int));
  cudaMemset((void **)&xMaxs,  0, N_ROWS * ncols * sizeof(int));
  cudaMemset((void **)&yMaxs,  0, N_ROWS * ncols * sizeof(int));

  dim3 g_h32x32(ceil(N_ROWS * ncols / 16.0), ceil(N_ROWS * ncols / 16.0));
  dim3 b_h32x32(16, 16);

  // hough_v0
  {
    cudaEventRecord(astartEvent, 0);

    hough_v0_0<<<g16x16, b16x16>>>(maskData, imageWidth, imageHeight, hArray, ncols, xMins, yMins, xMaxs, yMaxs);
    hough_v0_1<<<g_h32x32, b_h32x32>>>(hArray, ncols, 30);
    //foo<<<g_h32x32, b_h32x32>>>();
    cudaDeviceSynchronize();
    printf("Hello\n");

    cudaMemcpy(hArrayHost, hArray, N_ROWS * ncols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(xMinsHost, xMins, N_ROWS * ncols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(yMinsHost, xMins, N_ROWS * ncols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(xMaxsHost, xMins, N_ROWS * ncols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(yMaxsHost, xMins, N_ROWS * ncols * sizeof(int), cudaMemcpyDeviceToHost);

    hough_v0_2(hArrayHost, xMinsHost, yMinsHost, xMaxsHost, yMaxsHost, ncols, linesHost, &numLines);

    printf("\n%d ", numLines);

    cudaEventRecord(astopEvent, 0); cudaEventSynchronize(astopEvent); cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\nTotal compute time (ms) %f for hough_v0 \n", aelapsedTime);
  }

  // for (int i = 0; i < N_ROWS; i++) {

  //   printf("Row %d\n", i);
  //   for (int j = 0; j < ncols; j++) {
  //     printf("%d ", hArrayHost[i * ncols + j]);
  //   }
  //   printf("\n");
  // }

  cudaFree(hArray);
  cudaFree(xMins);
  cudaFree(yMins);
  cudaFree(xMaxs);
  cudaFree(yMaxs);

  free(linesHost);
  free(hArrayHost);
  free(xMinsHost);
  free(yMinsHost);
  free(xMaxsHost);
  free(yMaxsHost);


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
