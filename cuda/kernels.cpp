#include <iostream>
#include "../include/Coordinates.h"

extern "C" Coordinates* generateCoordinatesOnGPU(
    int width,
    int height,
    int x,
    double aspectRatio,
    Coordinates *output
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
