#pragma once
#include "Structs.h"

__device__ Vector vectorNormal(Vector vector);
__device__ Vector vectorAdd(Vector a, Vector b);
__device__ Vector vectorSubtract(Vector a, Vector b);
__device__ Vector vectorMultiply(Vector vector, float scalar);
__device__ Vector vectorCrossProduct(Vector a, Vector b);
__device__ double vectorDotProduct(Vector a, Vector b);

__device__ Vector triangleSurfaceNormal(Triangle triangle);
__device__ bool trianglePointIsWithin(Triangle triangle, Vector point);

__device__ Vector lineDirection(Line line);
__device__ Vector lineEvaluate(Line line, double t);
__device__ LineTriangleIntersection lineIntersectsTriangle(const Line &line, const Triangle &triangle);
__device__ LineTriangleIntersection lineIntersectsObject(const Line &line, Object object);
__device__ LineTriangleIntersection lineIntersectsLandscape(const Line &line, Landscape landscape);