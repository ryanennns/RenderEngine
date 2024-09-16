#include <iostream>
#include "../include/Structs.h"

extern "C" Coordinates *generateCoordinatesOnGPU(
    int width,
    int height,
    int x,
    double aspectRatio,
    Coordinates *output
);

extern "C" void determineLandscapeIntersectionsOnGPU(
    const Line *lines,
    const Landscape landscape,
    const int width,
    const int height,
    LineTriangleIntersection *objectIntersections
);

void generateCoordinatesForColumn(
    int width,
    int height,
    int x,
    double aspectRatio,
    Coordinates *coordinates
)
{
    generateCoordinatesOnGPU(width, height, x, aspectRatio, coordinates);
}

void determineLandscapeIntersections(
    const Line *lines,
    const Landscape landscape,
    const int width,
    const int height,
    LineTriangleIntersection *objectIntersections
)
{
    determineLandscapeIntersectionsOnGPU(
        lines,
        landscape,
        width,
        height,
        objectIntersections
    );
}
