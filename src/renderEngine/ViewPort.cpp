#include <iostream>
#include "../include/ViewPort.h"
#include "../include/kernels.h"

ViewPort::ViewPort()
{
    this->eye = Vector3D(0, 0, 0);
}

ViewPort::ViewPort(Vector3D eye, double width, double height)
{
    this->eye = eye;
}

Line *ViewPort::generateRays(const int width, const int height) const
{
    // Allocate a single array of Lines, treating it as a flattened 2D array
    Line *rays = new Line[width * height];

    const double aspectRatio = (double) width / (double) height;

    for (int x = 0; x < width; x++) {
        // Allocate coordinates for this column
        auto *coordinates = new Coordinates[height];

        // Call your generateCoordinatesForColumn function for this column
        generateCoordinatesForColumn(width, height, x, aspectRatio, coordinates);

        // Fill in the ray data for each row (height)
        for (int y = 0; y < height; y++) {
            constexpr double zOffset = 1;

            // Flattened index calculation
            int index = y * width + x;

            // Create the ray in the flattened array
            rays[index] = Line{
                Vector{
                    this->eye.x, this->eye.y, this->eye.z, // Eye position
                },
                Vector{
                    coordinates[y].x, coordinates[y].y, zOffset // Coordinates from generated points
                }
            };
        }

        // Clean up the coordinates array for the current column
        delete[] coordinates;
    }

    return rays; // Return the flattened 2D array of Line objects
}

vector<vector<Ray> > ViewPort::getRays()
{
    return this->rays;
}

Vector3D ViewPort::getEye()
{
    return this->eye;
}
