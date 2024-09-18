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
    const Landscape *landscape,
    const int width,
    const int height,
    LineTriangleIntersection *objectIntersections
)
{
    Vector line = lines[0].a;

    const unsigned int idx = blockIdx.x;
    const unsigned int idy = threadIdx.x;

    if (idx > width || idy > height) {
        return;
    }

    unsigned int index = idy * width + idx;
    LineTriangleIntersection intersect = lineIntersectsLandscape(lines[index], *landscape);

    objectIntersections[index] = intersect;
}

void printIfError(cudaError_t error)
{
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

__host__ Landscape *copyLandscapeToGPU(const Landscape landscape)
{
    Landscape *d_landscape = nullptr;
    auto error = cudaMalloc(&d_landscape, sizeof(Landscape));
    printIfError(error);

    Object *d_objects = nullptr;
    error = cudaMalloc(&d_objects, landscape.size * sizeof(Object));
    printIfError(error);

    for (int i = 0; i < landscape.size; i++) {
        const Object object = landscape.objects[i];

        Triangle *d_triangles = nullptr;
        error = cudaMalloc(&d_triangles, object.size * sizeof(Triangle));
        printIfError(error);

        for (int j = 0; j < object.size; j++) {
            const Triangle triangle = object.triangles[j];

            error = cudaMemcpy(&d_triangles[j], &triangle, sizeof(Triangle), cudaMemcpyHostToDevice);
            printIfError(error);
        }

        Object tmp_object = object;
        tmp_object.triangles = d_triangles;

        cudaMemcpy(&d_objects[i], &tmp_object, sizeof(Object), cudaMemcpyHostToDevice);
    }

    Landscape tmp_landscape = landscape;
    tmp_landscape.size = landscape.size;
    tmp_landscape.objects = d_objects;

    error = cudaMemcpy(d_landscape, &tmp_landscape, sizeof(Landscape), cudaMemcpyHostToDevice);
    printIfError(error);

    return d_landscape;
}

__host__ Line *copyLinesToGPU(const Line *lines, const int width, const int height)
{
    Line *d_lines = nullptr;
    const size_t size = width * height * sizeof(Line);

    cudaMalloc(&d_lines, size);
    cudaMemcpy(d_lines, lines, size, cudaMemcpyHostToDevice);

    return d_lines;
}

// __host__ void freeLinesFromGPU(const Line d_lines*, const int width, const int height)
// {
//     auto error = cudaFree((void *) &d_lines);
//     printIfError(error);
// }

__host__ void freeLandscapeFromGPU(
    const Landscape landscape,
    Landscape *d_landscape
)
{
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

    auto error = cudaMalloc(&d_output, size);
    printIfError(error);

    generateCoordinatesKernel<<<1, height>>>(width, height, x, aspectRatio, d_output);
    cudaDeviceSynchronize();

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    printIfError(error);

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

    const auto landscapePointer = copyLandscapeToGPU(landscape);
    const auto linesPointer = copyLinesToGPU(lines, width, height);
    // d_landscape is still null after this call
    cudaError_t error = cudaMalloc(&d_output, size);
    printIfError(error);

    intersectionKernel<<<width, height>>>(linesPointer, landscapePointer, width, height, d_output);
    error = cudaGetLastError();
    printIfError(error);

    error = cudaDeviceSynchronize();
    printIfError(error);

    error = cudaMemcpy(
        objectIntersections,
        d_output,
        size,
        cudaMemcpyDeviceToHost
    );
    printIfError(error);

    cudaFree(d_output);
}
