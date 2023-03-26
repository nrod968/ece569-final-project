#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <wb.h>
#include <png.h>

void save_image_to_ppm(const char* filename, wbImage_t image) {
    // Open the file for writing
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open file %s for writing\n", filename);
        return;
    }
  
    // Get the image width and height
    int width = wbImage_getWidth(image);
    int height = wbImage_getHeight(image);
  
    // Get the image data
    float* data = wbImage_getData(image);
  
    // Write the PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
  
    // Convert float values to unsigned char values
    unsigned char* converted_data = (unsigned char*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        converted_data[i * 3] = (unsigned char)(data[i * 3] * 255.0f);
        converted_data[i * 3 + 1] = (unsigned char)(data[i * 3 + 1] * 255.0f);
        converted_data[i * 3 + 2] = (unsigned char)(data[i * 3 + 2] * 255.0f);
    }
  
    // Write the pixel data
    fwrite(converted_data, 1, width * height * 3, fp);
  
    // Free the converted data
    free(converted_data);
  
    // Close the file
    fclose(fp);
  }
  
  void save_image_to_pgm(const char* filename, wbImage_t image) {
    // Open the file for writing
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open file %s for writing\n", filename);
        return;
    }
  
    // Get the image width and height
    int width = wbImage_getWidth(image);
    int height = wbImage_getHeight(image);
  
    // Get the image data
    float* data = wbImage_getData(image);
  
    // Write the PGM header
    fprintf(fp, "P2\n%d %d\n255\n", width, height);
  
    // Convert float values to integer values
    unsigned char* converted_data = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        converted_data[i] = (unsigned char)(data[i] * 255.0f);
    }
  
    // Write the pixel data
    for (int i = 0; i < width * height; i++) {
        fprintf(fp, "%d ", converted_data[i]);
    }
  
    // Free the converted data
    free(converted_data);
  
    // Close the file
    fclose(fp);
  }

  void save_image_to_pbm(const char* filename, wbImage_t image) {
    // Open the file for writing
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open file %s for writing\n", filename);
        return;
    }

    // Get the image width and height
    int width = wbImage_getWidth(image);
    int height = wbImage_getHeight(image);

    // Get the image data
    float* data = wbImage_getData(image);

    // Write the PBM header
    fprintf(fp, "P1\n%d %d\n", width, height);

    // Convert float values to binary values
    unsigned char* converted_data = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        converted_data[i] = (data[i] >= 0.5f) ? 1 : 0;
    }

    // Write the pixel data
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(fp, "%d ", converted_data[i * width + j]);
        }
        fprintf(fp, "\n");
    }

    // Free the converted data
    free(converted_data);

    // Close the file
    fclose(fp);
}