#pragma once
#include "Structs.h"

void generateCoordinatesForColumn(
    int width,
    int height,
    int x,
    double aspectRatio,
    Coordinates *coordinates
);

void determineLandscapeIntersections(
    const Line *lines,
    const Landscape landscape,
    const int width,
    const int height,
    LineTriangleIntersection *objectIntersections
);
