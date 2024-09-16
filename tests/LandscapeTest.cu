#include <helpers.h>
#include <gtest/gtest.h>
#include <ViewPort.h>
#include "../include/Structs.h"

TEST(LandscapeTest, it_detects_intersection_with_landscape)
{
    const auto line = Line{{0, 0, 0},{0, 0, 1}};

    const auto landscape = Landscape{
        1,
        new Object{
            1,
            new Triangle{
                {1.0f, 1.0f, 5.0f},
                {-1.0f, 1.0f, 5.0f},
                {0.0f, -1.0f, 5.0f}
            }
        }
    };

    const auto intersection = lineIntersectsLandscape(line, landscape);

    ASSERT_TRUE(intersection.intersects);
}