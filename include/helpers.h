#pragma once
#include "Structs.h"

__device__ Vector vectorNormal(Vector vector);
__device__ Vector vectorAdd(Vector a, Vector b);
__device__ Vector vectorSubtract(Vector a, Vector b);
__device__ Vector vectorMultiply(Vector vector, double scalar);
__device__ Vector vectorCrossProduct(Vector a, Vector b);
__device__ double vectorDotProduct(Vector a, Vector b);

__device__ Vector triangleSurfaceNormal(const Triangle &triangle);
__device__ bool trianglePointIsWithin(const Triangle &triangle, const Vector &point);

__device__ Vector lineDirection(const Line &line);
__device__ Vector lineEvaluate(const Line &line, double t);
__device__ LineTriangleIntersection lineIntersectsTriangle(const Line &line, const Triangle &triangle);
__device__ LineTriangleIntersection lineIntersectsObject(const Line &line, Object object);
__device__ LineTriangleIntersection lineIntersectsLandscape(const Line &line, Landscape landscape);