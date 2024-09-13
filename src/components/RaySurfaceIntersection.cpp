#include "../../include/RaySurfaceIntersection.h"
#include "../../include/Vector3D.h"
#include "../../include/Rgb.h"

RaySurfaceIntersection::RaySurfaceIntersection(
	Vector3D hitPoint,
	Vector3D surfaceNormal, 
	Rgb colour,
	double specularCoefficent)
{
	this->hitPoint = hitPoint;
	this->surfaceNormal = surfaceNormal;
	this->colour = colour;
	this->specularCoefficient = specularCoefficent;
}

Vector3D RaySurfaceIntersection::getHitPoint()
{
	return this->hitPoint;
}

Vector3D RaySurfaceIntersection::getNormal()
{
	return this->surfaceNormal;
}

Rgb RaySurfaceIntersection::getColour()
{
	return this->colour;
}

double RaySurfaceIntersection::getSpecularCoefficient()
{
	return this->specularCoefficient;
}