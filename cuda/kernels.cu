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
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = threadIdx.x;

    if (idx > width || idy > height) {
        return;
    }

    const unsigned int index = idy * width + idx;
    const LineTriangleIntersection intersect = lineIntersectsLandscape(lines[index], *landscape);

    objectIntersections[index] = intersect;
}

void printIfError(const cudaError_t error, const char* function, const int line)
{
    if (error != cudaSuccess) {
        printf("CUDA error at %s::%d: %s\n", function, line, cudaGetErrorString(error));
    }
}

__host__ Landscape *copyLandscapeToGPU(const Landscape landscape)
{
    Landscape *d_landscape = nullptr;
    auto error = cudaMalloc(&d_landscape, sizeof(Landscape));
    printIfError(error, __func__, __LINE__);

    Object *d_objects = nullptr;
    error = cudaMalloc(&d_objects, landscape.size * sizeof(Object));
    printIfError(error, __func__, __LINE__);

    for (int i = 0; i < landscape.size; i++) {
        const Object object = landscape.objects[i];

        Triangle *d_triangles = nullptr;
        error = cudaMalloc(&d_triangles, object.size * sizeof(Triangle));
        printIfError(error, __func__, __LINE__);

        for (int j = 0; j < object.size; j++) {
            const Triangle triangle = object.triangles[j];

            error = cudaMemcpy(&d_triangles[j], &triangle, sizeof(Triangle), cudaMemcpyHostToDevice);
            printIfError(error, __func__, __LINE__);
        }

        Object tmp_object = object;
        tmp_object.triangles = d_triangles;

        cudaMemcpy(&d_objects[i], &tmp_object, sizeof(Object), cudaMemcpyHostToDevice);
    }

    Landscape tmp_landscape = landscape;
    tmp_landscape.size = landscape.size;
    tmp_landscape.objects = d_objects;

    error = cudaMemcpy(d_landscape, &tmp_landscape, sizeof(Landscape), cudaMemcpyHostToDevice);
    printIfError(error, __func__, __LINE__);

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

__host__ void freeLinesFromGPU(const Line *d_lines)
{
    auto error = cudaFree((void *) d_lines);
    printIfError(error, __func__, __LINE__);
}

// TODO make this not sigsev
__host__ void freeLandscapeFromGPU(const Landscape landscape, Landscape *d_landscape)
{
    const Landscape h_landscape{};

    cudaMemcpy((void *) &h_landscape, d_landscape, sizeof(Landscape), cudaMemcpyDeviceToHost);
    for (int i = 0; i < landscape.size; i++) {
        const Object object = landscape.objects[i];
        const Object d_object{};

        cudaMemcpy((void *) &d_object, &h_landscape.objects[i], sizeof(Object), cudaMemcpyDeviceToHost);
        for (int j = 0; j < object.size; j++) {
            const Triangle d_triangle{};
            cudaMemcpy((void *) &d_triangle, &d_object.triangles[j], sizeof(Triangle), cudaMemcpyDeviceToHost);

            cudaFree(&d_object.triangles[j]);
        }

        cudaFree(&landscape.objects[i]);
    }
    cudaFree(d_landscape);
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
    printIfError(error, __func__, __LINE__);

    generateCoordinatesKernel<<<1, height>>>(width, height, x, aspectRatio, d_output);
    cudaDeviceSynchronize();

    error = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    printIfError(error, __func__, __LINE__);

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

    const auto d_landscape = copyLandscapeToGPU(landscape);
    const auto d_lines = copyLinesToGPU(lines, width, height);

    cudaError_t error = cudaMalloc(&d_output, size);
    printIfError(error, __func__, __LINE__);

    intersectionKernel<<<width, height>>>(d_lines, d_landscape, width, height, d_output);
    error = cudaGetLastError();
    printIfError(error, __func__, __LINE__);

    error = cudaDeviceSynchronize();
    printIfError(error, __func__, __LINE__);

    error = cudaMemcpy(
        objectIntersections,
        d_output,
        size,
        cudaMemcpyDeviceToHost
    );
    printIfError(error, __func__, __LINE__);

    freeLandscapeFromGPU(landscape, d_landscape);
    freeLinesFromGPU(d_lines);
    cudaFree(d_output);
}
