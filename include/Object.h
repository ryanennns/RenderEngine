#pragma once

#include <vector>
#include <memory>

#include "Surface.h"
#include "Ray.h"
#include "RaySurfaceIntersection.h"

class Object
{
public:
	Object();
	Object(Object&);
	~Object();
	std::vector<RaySurfaceIntersection> intersections(Ray ray);
	void addSurface(Surface* surface);
	void transform(Vector3D translation, Vector3D rotation = Vector3D(0,0,0));
	void translate(Vector3D translation);
	void rotate(Vector3D rotation);
	void setColour(Rgb colour);
	Rgb getColour();
private:
	std::vector<Surface*> surfaces;
	Rgb colour;
};