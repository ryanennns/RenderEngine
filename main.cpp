#include <iostream>
#include <chrono>
#include <SDL2/SDL.h>
#include "include/ViewPort.h"
#include <kernels.h>
#include <Structs.h>

int main()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    constexpr int width = 500;
    constexpr int height = 500;

    SDL_Window *window = SDL_CreateWindow("Landscape Intersections", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          width, height, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    const auto viewport = ViewPort();

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        auto start = std::chrono::high_resolution_clock::now();

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

        Landscape landscape = {
            numberOfObjects,
            objects
        };

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
        printf("\rREFRESH RATE: %.1lfhz\n", 1 / (ms * 0.001));
        fflush(stdout);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                const auto &intersection = objectIntersections[index];

                if (intersection.intersects) {
                    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
                } else {
                    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                }

                SDL_RenderDrawPoint(renderer, x, y);
            }
        }


        SDL_RenderPresent(renderer);


        for (int i = 0; i < numberOfObjects; ++i) {
            delete[] objects[i].triangles;
        }
        delete[] rays;
        delete[] objectIntersections;
    }


    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
