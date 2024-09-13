#pragma once

#include <vector>
#include "Rgb.h"
#include "Ray.h"
#include "Vector3D.h"

class Surface
{
protected:
	Rgb colour;
public:
	virtual ~Surface() {};
	virtual std::vector<Vector3D> intersections(Ray) = 0;
	virtual void transform(
		Vector3D translation,
		Vector3D rotation = Vector3D(0, 0, 0)
	) = 0;
	virtual void translate(Vector3D) = 0;
	virtual void rotate(Vector3D) = 0;
	virtual Surface* clone() const = 0;
	virtual Rgb getColour() = 0;
	virtual void setColour(Rgb) = 0;
	virtual bool verifyIntersection(Vector3D, Vector3D) = 0;
	virtual Vector3D getNormal(Vector3D) = 0;
	virtual double getSpecularCoefficient() = 0;
};