#include <cmath>
#include "../../../include/Triangle.h"
#include "../../../include/Vector3D.h"
#include "../../../include/Ray.h"

Triangle::Triangle(Vector3D A, Vector3D B, Vector3D C)
{
	this->A = A;
	this->B = B;
	this->C = C;
	this->colour = Rgb(128, 0, 0);
	this->specularCoefficient = 2;
}

Triangle::Triangle(Vector3D A, Vector3D B, Vector3D C, Rgb colour)
{
	this->A = A;
	this->B = B;
	this->C = C;
	this->colour = colour;
	this->specularCoefficient = 2;
}

Triangle::Triangle(Vector3D A, Vector3D B, Vector3D C, Rgb colour, double specularCoefficient)
{
	this->A = A;
	this->B = B;
	this->C = C;
	this->colour = colour;
	this->specularCoefficient = specularCoefficient;
}

Vector3D Triangle::getA()
{
	return this->A;
}

Vector3D Triangle::getB()
{
	return this->B;
}

Vector3D Triangle::getC()
{
	return this->C;
}

Rgb Triangle::getColour()
{
	return this->colour;
}

void Triangle::setA(Vector3D A)
{
	this->A = A;
}

void Triangle::setB(Vector3D B)
{
	this->B = B;
}

void Triangle::setC(Vector3D C)
{
	this->C = C;
}

void Triangle::setColour(Rgb colour)
{
	this->colour = colour;
}

Vector3D Triangle::normal()
{
	Vector3D AB = B.subtract(A);
	Vector3D AC = C.subtract(A);
	Vector3D normal = AB.crossProduct(AC);
	normal.normalize();
	return normal;
}

std::vector<Vector3D> Triangle::intersections(Ray ray)
{
	const double epsilon = 1e-12;
	std::vector<Vector3D> returnVector;

	Vector3D normal = this->normal();
	double NdotRayDirection = normal.dotProduct(ray.getDirection());

	if (NdotRayDirection == 0.0) {
		return returnVector;
	}

	double d = -normal.dotProduct(this->getA());
	double t = -((normal.dotProduct(ray.getOrigin())) + d)
				/ NdotRayDirection;

	Vector3D planeIntersection = ray.evaluate(t);
	if (
		this->isPointInTriangle(planeIntersection)
		&& t > 0
		&& this->verifyIntersection(planeIntersection, ray.getOrigin())
	) {
		returnVector.push_back(planeIntersection);
	}

	return returnVector;
}

bool Triangle::isPointInTriangle(Vector3D point)
{
	Vector3D edge0 = B.subtract(A);
	Vector3D edge1 = C.subtract(B);
	Vector3D edge2 = A.subtract(C);

	Vector3D C0 = point.subtract(A);
	Vector3D C1 = point.subtract(B);
	Vector3D C2 = point.subtract(C);

	if (this->normal().dotProduct(edge0.crossProduct(C0)) > 0 &&
		this->normal().dotProduct(edge1.crossProduct(C1)) > 0 &&
		this->normal().dotProduct(edge2.crossProduct(C2)) > 0) {

		return true;
	}

	return false;
}

void Triangle::transform(Vector3D translation, Vector3D rotation)
{
	this->A = this->A.add(translation);
	this->B = this->B.add(translation);
	this->C = this->C.add(translation);
}

void Triangle::translate(Vector3D translation)
{
	this->A = this->A.add(translation);
	this->B = this->B.add(translation);
	this->C = this->C.add(translation);
}

void Triangle::rotate(Vector3D rotation)
{
	// todo -- implement rotation
}

Surface* Triangle::clone() const
{
	return new Triangle(*this);
}

bool Triangle::verifyIntersection(Vector3D a, Vector3D b)
{
	// todo -- revisit this in case it causes problems down the line
	return std::fabs(a.x - b.x) > 1e-12
		|| std::fabs(a.y - b.y) > 1e-12
		|| std::fabs(a.z - b.z) > 1e-12;
}

Vector3D Triangle::getNormal(Vector3D point)
{
	return this->normal();
}

double Triangle::getSpecularCoefficient()
{
	return this->specularCoefficient;
}