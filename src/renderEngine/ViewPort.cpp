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

vector<vector<Line> > ViewPort::generateRays(const int width, const int height) const
{
    vector<vector<Line> > rays;
    const double aspectRatio = (double) width / (double) height;

    for (int x = 0; x < width; x++) {
        auto *coordinates = new Coordinates[height];
        generateCoordinatesForColumn(width, height, x, aspectRatio, coordinates);

        vector<Line> newRays;
        for (int i = 0; i < height; i++) {
            const double zOffset = 1;
            newRays.push_back(Line{
                Vector{
                    this->eye.x, this->eye.y, this->eye.z,
                },
                Vector{coordinates[i].x, coordinates[i].y, zOffset}
            });
        }
        rays.push_back(newRays);

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
