#pragma once

#include "Surface.h"
#include "Vector3D.h"
#include "Ray.h"

#include <vector>

class Triangle : public Surface
{
private:
	Vector3D A, B, C;

	bool isPointInTriangle(Vector3D);
	double specularCoefficient;
public:
	//static const double TRIANGLE_EPSILON = 1e-12;

	Triangle(Vector3D, Vector3D, Vector3D);
	Triangle(Vector3D, Vector3D, Vector3D, Rgb);
	Triangle(Vector3D, Vector3D, Vector3D, Rgb, double);

	Vector3D getA();
	Vector3D getB();
	Vector3D getC();

	void setA(Vector3D);
	void setB(Vector3D);
	void setC(Vector3D);

	Vector3D normal();

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