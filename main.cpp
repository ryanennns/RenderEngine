#include <iostream>
#include "include/ViewPort.h"
#include <chrono>
#include <kernels.h>
#include <Structs.h>

int main()
{
    const auto viewport = ViewPort();

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        constexpr int numberOfObjects = 3;
        Object objects[numberOfObjects] = {
            Object{
                1,
                new Triangle[1]{
                    Triangle{
                        Vector{0.0f, 0.0f, 0.0f},
                        Vector{1.0f, 0.0f, 0.0f},
                        Vector{0.0f, 1.0f, 0.0f}
                    }
                }
            },
            Object{
                1,
                new Triangle[1]{
                    Triangle{
                        Vector{1.0f, 0.0f, 0.0f},
                        Vector{2.0f, 0.0f, 1.0f},
                        Vector{1.0f, 1.0f, 0.0f}
                    }
                }
            },
            Object{
                1,
                new Triangle[1]{
                    Triangle{
                        Vector{0.0f, 1.0f, 0.0f},
                        Vector{1.0f, 2.0f, 1.0f},
                        Vector{0.0f, 2.0f, 0.0f}
                    }
                }
            }
        };

        Landscape landscape = {
            3,
            objects
        };

        constexpr int width = 1920;
        constexpr int height = 1080;

        const Line *rays = viewport.generateRays(width, height);

        auto *objectIntersections = new LineTriangleIntersection[width * height];
        determineLandscapeIntersections(
            rays,
            landscape,
            width,
            height,
            objectIntersections
        );

        auto finish = std::chrono::high_resolution_clock::now();
        int ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        printf("REFRESH RATE: %.1lf/s\n", 1 / (ms * 0.001));

        for (int i = 0; i < numberOfObjects; ++i) {
            delete[] objects[i].triangles;
        }
        delete[] rays;
        delete[] objectIntersections;
    }
}
