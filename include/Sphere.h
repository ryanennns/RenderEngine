#pragma once


#include "Surface.h"
#include "Vector3D.h"
#include "Ray.h"
#include <vector>

class Sphere : public Surface
{
private:
	Vector3D center;
	double radius;
	double specularCoefficient;

	double intersectDiscriminant(Ray);
public:
	//static const double SPHERE_EPSILON = 1e-12;

	Sphere();
	Sphere(const Vector3D&, double);
	Sphere(const Vector3D&, double, Rgb);
	Sphere(const Vector3D&, double, Rgb, double);
	Sphere(const Sphere&);

	Vector3D getCenter();
	double getRadius();

	std::vector<Vector3D> intersections(Ray) override;
	void transform(Vector3D translation, Vector3D rotation = Vector3D(0, 0, 0)) override;
	void translate(Vector3D) override;
	void rotate(Vector3D) override;
	Surface* clone() const override;

	Rgb getColour() override;
	void setColour(Rgb) override;

	bool verifyIntersection(Vector3D, Vector3D) override;
	Vector3D getNormal(Vector3D) override;

	double getSpecularCoefficient() override;
};