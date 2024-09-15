#include <iostream>
#include "include/ViewPort.h"
#include <chrono>

int main()
{
    auto viewport = ViewPort();

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        vector<vector<Ray> > rays = viewport.generateRays(10.0, 5.0);

        auto finish = std::chrono::high_resolution_clock::now();
        int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        printf("RAY GENERATION RATE: %.1lf/s\n", 1 / (ms * 0.001));
    }
}
