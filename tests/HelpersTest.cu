#include <gtest/gtest.h>
#include "../include/Structs.h"
#include "../include/helpers.h"

TEST(LineTriangleIntersectionTest, it_detects_intersection_with_triangle) {
    constexpr Line line = { {0, 0, 0}, {0, 0, 5} };
    constexpr Triangle triangle = { {1, 1, 3}, {-1, 1, 3}, {0, -1, 3} };

    constexpr Vector expectedIntersection = {0, 0, 3};

    LineTriangleIntersection result = lineIntersectsTriangle(line, triangle);

    EXPECT_TRUE(result.intersects);
    EXPECT_EQ(result.intersection, expectedIntersection);
}

TEST(LineTriangleIntersectionTest, test_it_detects_lack_of_intersection) {
    constexpr Line line = { {0, 0, 0}, {0, 0, -5} };
    constexpr Triangle triangle = { {1, 1, 3}, {-1, 1, 3}, {0, -1, 3} };

    const LineTriangleIntersection result = lineIntersectsTriangle(line, triangle);

    EXPECT_FALSE(result.intersects);
}