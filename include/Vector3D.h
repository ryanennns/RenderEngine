#pragma once

class Vector3D {
public:
    double x, y, z;

    Vector3D();
    Vector3D(Vector3D* v);
    Vector3D(double x, double y, double z);

    double magnitude();
    Vector3D normalize();
    Vector3D add(const Vector3D& v);
    Vector3D subtract(const Vector3D& v);
    Vector3D multiply(double scalar);
    double dotProduct(const Vector3D& v);
    Vector3D crossProduct(const Vector3D& v);
    double distanceBetween(const Vector3D& v);
    void consoleDisplay();
    Vector3D negative();
};