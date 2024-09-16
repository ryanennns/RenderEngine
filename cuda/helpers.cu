#include <cstdio>

#include "../include/Structs.h"

__device__ __host__ Vector vectorNormal(const Vector vector)
{
    const float magnitude = sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);

    return Vector{
        vector.x / magnitude,
        vector.y / magnitude,
        vector.z / magnitude
    };
}

__device__ __host__ Vector vectorAdd(Vector a, Vector b)
{
    return Vector{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

__device__ __host__ Vector vectorSubtract(Vector a, Vector b)
{
    return Vector{
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

__device__ __host__ Vector vectorMultiply(Vector vector, double scalar)
{
    return Vector{
        vector.x * scalar,
        vector.y * scalar,
        vector.z * scalar
    };
}

__device__ __host__ Vector vectorCrossProduct(Vector a, Vector b)
{
    return Vector{
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.x)
    };
}

__device__ __host__ double vectorDotProduct(Vector a, Vector b)
{
    return (
        (a.x * b.x) +
        (a.y * b.y) +
        (a.z * b.z)
    );
}

__device__ __host__ Vector triangleSurfaceNormal(const Triangle &triangle)
{
    const Vector AB = vectorSubtract(triangle.b, triangle.b);
    const Vector AC = vectorSubtract(triangle.c, triangle.a);

    const Vector normal = vectorCrossProduct(AB, AC);
    vectorNormal(normal);

    return normal;
}

__device__ __host__ bool trianglePointIsWithin(const Triangle &triangle, const Vector &point)
{
    const Vector edge0 = vectorSubtract(triangle.b, triangle.a);
    const Vector edge1 = vectorSubtract(triangle.c, triangle.b);
    const Vector edge2 = vectorSubtract(triangle.a, triangle.c);

    const Vector C0 = vectorSubtract(point, triangle.a);
    const Vector C1 = vectorSubtract(point, triangle.b);
    const Vector C2 = vectorSubtract(point, triangle.c);

    const Vector triangleNormal = triangleSurfaceNormal(triangle);

    if (
        vectorDotProduct(triangleNormal, vectorCrossProduct(edge0, C0)) > 0 &&
        vectorDotProduct(triangleNormal, vectorCrossProduct(edge1, C1)) > 0 &&
        vectorDotProduct(triangleNormal, vectorCrossProduct(edge2, C2)) > 0
    ) {
        return true;
    }

    return false;
}

__device__ __host__ Vector lineDirection(const Line &line)
{
    return vectorSubtract(line.b, line.a);
}

__device__ __host__ Vector lineEvaluate(const Line &line, const double t)
{
    return vectorAdd(line.a, vectorMultiply(lineDirection(line), t));
}

__device__ __host__ LineTriangleIntersection lineIntersectsTriangle(const Line &line, const Triangle &triangle)
{
    const auto normal = triangleSurfaceNormal(triangle);
    const double nDotRayDirection = vectorDotProduct(normal, lineDirection(line));

    if (nDotRayDirection < 0) {
        return LineTriangleIntersection{
            false,
            triangle,
            line,
            Vector{-1, -1, -1},
        };
    }

    const double d = -vectorDotProduct(normal, triangle.a);
    const double t = -(vectorDotProduct(normal, line.a) + d) / nDotRayDirection;

    const Vector planeIntersection = lineEvaluate(line, t);

    if (
        trianglePointIsWithin(triangle, planeIntersection)
        && t > 0
        // && this->verifyIntersection(planeIntersection, ray.getOrigin())
    ) {
        return LineTriangleIntersection{
            true,
            triangle,
            line,
            planeIntersection,
        };
    }

    return LineTriangleIntersection{
        false,
        triangle,
        line,
        planeIntersection,
    };
}

__device__ __host__ LineTriangleIntersection lineIntersectsObject(const Line &line, const Object object)
{
    for (int i = 0; i < object.size; i++) {
        const auto intersection = lineIntersectsTriangle(line, object.triangles[i]);
        if (intersection.intersects) {
            return intersection;
        }
    }

    return LineTriangleIntersection{
        false,
        Triangle{Vector{0, 0, 0}, Vector{0, 0, 0}, Vector{0, 0, 0}},
        line,
        Vector{0, 0, 0}
    };
}

__device__ __host__ LineTriangleIntersection lineIntersectsLandscape(const Line &line, const Landscape landscape)
{
    for (int i = 0; i < landscape.size; i++) {
        const Object object = landscape.objects[i];

        const LineTriangleIntersection intersection = lineIntersectsObject(line, object);
        if (intersection.intersects) {
            return intersection;
        }
    }

    return LineTriangleIntersection{
        false,
        Triangle{Vector{0, 0, 0}, Vector{0, 0, 0}, Vector{0, 0, 0}},
        Line{Vector{0, 0, 0}, Vector{0, 0, 0}},
        Vector{0, 0, 0},
    };
}
