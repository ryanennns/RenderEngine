#pragma once

#include "Vector3D.h"
#include "Object.h"

class HitDetection
{
public:
	HitDetection(Vector3D, Vector3D, Rgb, double);

	Vector3D getHitPoint();
	Vector3D getNormal();
	Rgb getColour();
	double getSpecularCoefficient();
	void setSpecularCoefficient(double);
private:
	Vector3D hitPoint;
	Vector3D surfaceNormal;
	Rgb colour;
	double specularCoefficient;
};