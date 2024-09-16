#pragma once
#include "Structs.h"

__device__ __host__ Vector vectorNormal(Vector vector);
__device__ __host__ Vector vectorAdd(Vector a, Vector b);
__device__ __host__ Vector vectorSubtract(Vector a, Vector b);
__device__ __host__ Vector vectorMultiply(Vector vector, double scalar);
__device__ __host__ Vector vectorCrossProduct(Vector a, Vector b);
__device__ __host__ double vectorDotProduct(Vector a, Vector b);

__device__ __host__ Vector triangleSurfaceNormal(const Triangle &triangle);
__device__ __host__ bool trianglePointIsWithin(const Triangle &triangle, const Vector &point);

__device__ __host__ Vector lineDirection(const Line &line);
__device__ __host__ Vector lineEvaluate(const Line &line, double t);
__device__ __host__ LineTriangleIntersection lineIntersectsTriangle(const Line &line, const Triangle &triangle);
__device__ __host__ LineTriangleIntersection lineIntersectsObject(const Line &line, Object object);
__device__ __host__ LineTriangleIntersection lineIntersectsLandscape(const Line &line, Landscape landscape);