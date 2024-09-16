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
    const auto rays = new Line[width * height];

    const double aspectRatio = (double) width / (double) height;

    for (int x = 0; x < width; x++) {
        auto *coordinates = new Coordinates[height];

        generateCoordinatesForColumn(width, height, x, aspectRatio, coordinates);

        for (int y = 0; y < height; y++) {
            constexpr double zOffset = 1;

            int index = y * width + x;

            rays[index] = Line{
                Vector{
                    this->eye.x, this->eye.y, this->eye.z,
                },
                Vector{
                    coordinates[y].x, coordinates[y].y, zOffset
                }
            };
        }

        delete[] coordinates;
    }

    return rays;
}

vector<vector<Ray> > ViewPort::getRays()
{
    return this->rays;
}

Vector3D ViewPort::getEye()
{
    return this->eye;
}
