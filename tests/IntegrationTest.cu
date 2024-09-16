#include <gtest/gtest.h>
#include <kernels.h>
#include <ViewPort.h>
#include "../include/Structs.h"

TEST(IntegrationTest, it_detects_intersection_with_rays_and_triangles)
{
    const int width = 1;
    const int height = 1;

    const auto viewport = ViewPort();
    const Line *lines = viewport.generateRays(width, height);
    constexpr int numberOfObjects = 1;
    Object objects[numberOfObjects] = {
        Object{
            1,
            new Triangle[1]{
                Triangle{
                    Vector{1.0f, 1.0f, 5.0f},
                    Vector{-1.0f, 1.0f, 5.0f},
                    Vector{0.0f, -1.0f, 5.0f}
                }
            }
        }
    };

    const Landscape landscape = {
        numberOfObjects,
        objects
    };

    auto *objectIntersections = new LineTriangleIntersection[width * height];
    determineLandscapeIntersections(
        lines,
        landscape,
        width,
        height,
        objectIntersections
    );

    ASSERT_EQ(objectIntersections[0].intersects, true);
}
