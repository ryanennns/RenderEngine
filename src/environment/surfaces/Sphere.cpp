#include "../../../include/Sphere.h"
#include "stdio.h"
#include <cmath>
#include <vector>

using namespace std;

Sphere::Sphere()
{
	this->center = Vector3D(0,0,0);
	this->radius = 1;
	this->colour = Rgb(0, 0, 0);
	this->specularCoefficient = 2;
}

Sphere::Sphere(const Vector3D& center, double radius)
{
	this->center = Vector3D(center);
	this->radius = radius;
	this->colour = Rgb(0, 0, 0);
	this->specularCoefficient = 2;
}

Sphere::Sphere(const Vector3D& center, double radius, Rgb colour)
{
	this->center = Vector3D(center);
	this->radius = radius;
	this->colour = colour;
	this->specularCoefficient = 2;
}

Sphere::Sphere(const Sphere& sphere)
{
	this->center = Vector3D(sphere.center);
	this->radius = sphere.radius;
	this->colour = sphere.colour;
	this->specularCoefficient = sphere.specularCoefficient;
}

Sphere::Sphere(const Vector3D& center, double radius, Rgb colour, double specularCoefficient)
{
	this->center = Vector3D(center);
	this->radius = radius;
	this->colour = colour;
	this->specularCoefficient = specularCoefficient;
}

Vector3D Sphere::getCenter()
{
	return this->center;
}

double Sphere::getRadius()
{
	return this->radius;
}

Rgb Sphere::getColour()
{
	return this->colour;
}

void Sphere::setColour(Rgb colour)
{
	this->colour = colour;
}

std::vector<Vector3D> Sphere::intersections(Ray ray)
{
	const double epsilon = 1e-12;
	const double discriminant = this->intersectDiscriminant(ray);
	vector<Vector3D> returnVector;

	if (discriminant < 0.0) {
		return returnVector;
	}

	Vector3D d = ray.getDirection();
	Vector3D e = ray.getOrigin();
	Vector3D center = this->center;

	double a = d.dotProduct(d);
	double b = 2 * d.dotProduct(e.subtract(center));

	double t1 = (-b - sqrt(discriminant)) / (2 * a);

	Vector3D firstIntersect = ray.evaluate(t1);
	if (t1 > 0.0 && this->verifyIntersection(firstIntersect, ray.getOrigin()))
	{
		returnVector.push_back(firstIntersect);
	}

	if (discriminant > 0.0 && (-b + sqrt(discriminant)) / (2 * a) > 0.0) {
		double t2 = (-b + sqrt(discriminant)) / (2 * a);

		Vector3D secondIntersect = ray.evaluate(t2);

		if (this->verifyIntersection(secondIntersect, ray.getOrigin()))
		{
			returnVector.push_back(secondIntersect);
		}
	}

	return returnVector;
}

double Sphere::intersectDiscriminant(Ray ray)
{
	Vector3D d = ray.getDirection();
	Vector3D e = ray.getOrigin();
	Vector3D center = this->center;
	double radius = this->radius;

	double a = d.dotProduct(d);
	double b = 2 * (d.dotProduct(e.subtract(center)));
	double c = e.subtract(center).dotProduct(e.subtract(center)) - pow(radius, 2);

	return pow(b, 2) - (4.0 * (a * c));
}

void Sphere::transform(Vector3D translation, Vector3D rotation)
{
	this->center = this->center.add(translation);
}

void Sphere::translate(Vector3D translation)
{
	this->center = this->center.add(translation);
}

void Sphere::rotate(Vector3D rotation)
{
	// todo -- implement rotation
}

Surface* Sphere::clone() const
{
	return new Sphere(*this);
}

bool Sphere::verifyIntersection(Vector3D a, Vector3D b)
{
	// todo -- same here
	return std::fabs(a.x - b.x) > 1e-12
		|| std::fabs(a.y - b.y) > 1e-12
		|| std::fabs(a.z - b.z) > 1e-12;
}

Vector3D Sphere::getNormal(Vector3D point)
{
	return point.subtract(this->center).normalize();
}

double Sphere::getSpecularCoefficient()
{
	return this->specularCoefficient;
}