#pragma once

#include "Vector3D.h"
#include "Rgb.h"

class RaySurfaceIntersection
{
public:
	RaySurfaceIntersection(Vector3D, Vector3D, Rgb, double);

	Vector3D getHitPoint();
	Vector3D getNormal();
	Rgb getColour();
	double getSpecularCoefficient();
private:
	Vector3D hitPoint;
	Vector3D surfaceNormal;
	Rgb colour;
	double specularCoefficient;
};