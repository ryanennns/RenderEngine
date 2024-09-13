#include "../../include/HitDetection.h"
#include "../../include/Vector3D.h"
#include "../../include/Object.h"

HitDetection::HitDetection(Vector3D hitPoint, Vector3D surfaceNormal, Rgb colour, double specularCoefficent)
{
	this->hitPoint = hitPoint;
	this->surfaceNormal = surfaceNormal;
	this->colour = colour;
	this->specularCoefficient = specularCoefficent;
}

Vector3D HitDetection::getHitPoint()
{
	return this->hitPoint;
}

Rgb HitDetection::getColour()
{
	return this->colour;
}

Vector3D HitDetection::getNormal()
{
	return this->surfaceNormal;
}

double HitDetection::getSpecularCoefficient()
{
	return this->specularCoefficient;
}

void HitDetection::setSpecularCoefficient(double specularCoefficient)
{
	this->specularCoefficient = specularCoefficient;
}