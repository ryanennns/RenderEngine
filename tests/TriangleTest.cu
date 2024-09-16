#include <gtest/gtest.h>
#include "../include/Structs.h"
#include "../include/helpers.h"

TEST(TriangleIntersectionTest, it_detects_intersection_with_triangle)
{
    const Line line = {{0, 0, 0}, {0, 0, 1}};
    const Triangle triangle = {
        {5, 0, 3},
        {-5, 5, 3},
        {-5, -5, 3}
    };

    const Vector expectedIntersection = {0, 0, 3};

    const LineTriangleIntersection result = lineIntersectsTriangle(line, triangle);

    EXPECT_EQ(result.intersection, expectedIntersection);
    EXPECT_TRUE(result.intersects);
}

TEST(TriangleIntersectionTest, test_it_detects_lack_of_intersection)
{
    const Line line = {{0, 0, 0}, {0, 0, -5}};
    const Triangle triangle = {{1, 1, 3}, {-1, 1, 3}, {0, -1, 3}};

    const LineTriangleIntersection result = lineIntersectsTriangle(line, triangle);

    EXPECT_FALSE(result.intersects);
}

TEST(TriangleIntersectionTest, test_it_determines_if_point_is_above_or_below_triangle)
{
    const Triangle triangle = {{1, 1, 3}, {-1, 1, 3}, {0, -1, 3}};
    const Vector vector = {0, 0, 5};

    EXPECT_TRUE(trianglePointIsWithin(triangle, vector));
}

TEST(TriangleIntersectionTest, test_it_determines_triangle_surface_normal)
{
    const Triangle triangle = {{1, 1, 3}, {-1, 1, 3}, {0, -1, 3}};
    const Vector expected = {0, 0, 1};

    const Vector actual = triangleSurfaceNormal(triangle);

    EXPECT_EQ(expected, actual);
}
