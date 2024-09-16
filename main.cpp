#include <iostream>
#include "include/ViewPort.h"
#include <chrono>
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

        constexpr int width = 100;
        constexpr int height = 200;

        vector<vector<Line> > rays = viewport.generateRays(width, height);

        auto finish = std::chrono::high_resolution_clock::now();
        int ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        printf("RAY GENERATION RATE: %.1lf/s\n", 1 / (ms * 0.001));


        for (int i = 0; i < numberOfObjects; ++i) {
            delete[] objects[i].triangles;
        }
    }
}
