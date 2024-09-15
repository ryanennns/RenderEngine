#include "../include/Coordinates.h"
#include "stdio.h"
#include "cuda_runtime.h"

// CUDA kernel that operates on primitives and outputs coordinates
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

    // Copy the result back to the host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
}
