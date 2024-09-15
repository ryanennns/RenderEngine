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

vector<vector<Ray> > ViewPort::generateRays(int width, int height)
{
    vector<vector<Ray> > rays;
    const double aspectRatio = (double) width / (double) height;
    double zOffset = 1;

    for (int x = 0; x < width; x++) {
        auto *coordinates = new Coordinates[height];
        generateCoordinatesForColumn(width, height, x, aspectRatio, coordinates);

        vector<Ray> newRays;
        for (int i = 0; i < height; i++) {
            newRays.push_back(Ray(this->eye, Vector3D(coordinates[i].x, coordinates[i].y, zOffset)));
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
