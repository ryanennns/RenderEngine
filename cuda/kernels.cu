#include "../include/Structs.h"
#include "../include/helpers.h"
#include <cstdio>
#include "cuda_runtime.h"

__global__ void generateCoordinatesKernel(
    const int width,
    const int height,
    const int x,
    const double aspectRatio,
    Coordinates *output
)
{
    const unsigned int idx = threadIdx.x;
    if (idx < width) {
        double normalizedX = (idx + 0.5) / width;
        double normalizedY = (x + 0.5) / height;

        normalizedX = (2 * normalizedX) - 1;
        normalizedY = (2 * normalizedY) - 1;

        normalizedX *= aspectRatio;

        output[idx] = {normalizedX, normalizedY};
    }
}

__global__ void intersectionKernel(
    const Line *lines,
    const Landscape landscape,
    const int width,
    const int height,
    LineTriangleIntersection *objectIntersections
)
{
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = threadIdx.x;

    if (idx > width || idy > height) {
        return;
    }

    unsigned int index = idy * width + idx;
    LineTriangleIntersection intersect = lineIntersectsLandscape(lines[index], landscape);

    objectIntersections[index] = intersect;
}


extern "C" void generateCoordinatesOnGPU(
    const int width,
    const int height,
    const int x,
    const double aspectRatio,
    Coordinates *output
)
{
    Coordinates *d_output = nullptr;
    const size_t size = width * height * sizeof(Coordinates);

    cudaMalloc((void **) &d_output, size);

    generateCoordinatesKernel<<<1, height>>>(width, height, x, aspectRatio, d_output);
    cudaDeviceSynchronize();

    const auto error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        printf("CUDA memcpy error in coordinate generation: %s\n", cudaGetErrorString(error));
    }

    cudaFree(d_output);
}

extern "C" void determineLandscapeIntersectionsOnGPU(
    const Line *lines,
    const Landscape landscape,
    const int width,
    const int height,
    LineTriangleIntersection *objectIntersections
)
{
    LineTriangleIntersection *d_output = nullptr;
    const size_t size = width * height * sizeof(LineTriangleIntersection);

    cudaError_t error = cudaMalloc((void **) &d_output, size);

    if (error != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(error));
    }

    intersectionKernel<<<width, height>>>(lines, landscape, width, height, d_output);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA memcpy error: %s\n", cudaGetErrorString(error));
        std::exit(-1);
    }

    error = cudaMemcpy(
        objectIntersections,
        d_output,
        size,
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        printf("CUDA memcpy error: %s\n", cudaGetErrorString(error));
        // std::exit(-1);
    }

    cudaFree(d_output);
}
