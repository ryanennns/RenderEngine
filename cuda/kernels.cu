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
    LineTriangleIntersection *objectIntersections
)
{
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = threadIdx.x;

    LineTriangleIntersection intersect = lineIntersectsLandscape(lines[idy * width + idx], landscape);

    objectIntersections[idy * width + idx] = intersect;
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
    const size_t size = height * sizeof(Coordinates);

    cudaMalloc((void **) &d_output, size);

    generateCoordinatesKernel<<<1, height>>>(width, height, x, aspectRatio, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

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

    cudaMalloc((void **) &d_output, size);

    intersectionKernel<<<width, height>>>(lines, landscape, width, d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(
        objectIntersections,
        d_output,
        size,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_output);
}
