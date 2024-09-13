#include "../../include/Object.h"
#include "../../include/Ray.h"
#include "../../include/RaySurfaceIntersection.h"

using namespace std;

Object::Object()
{
	this->colour = Rgb(0, 255, 0);
}

Object::Object(Object& object)
{
	this->colour = Rgb(0, 0, 0);
	for (int i = 0; i < object.surfaces.size(); i++)
	{
		this->addSurface(object.surfaces.at(i)->clone());
	}
}

Object::~Object()
{
	for (int i = 0; i < surfaces.size(); i++)
	{
		delete surfaces.at(i);
	}
}

vector<RaySurfaceIntersection> Object::intersections(Ray ray)
{
	vector<RaySurfaceIntersection> returnVector;

	for (int i = 0; i < surfaces.size(); i++)
	{
		vector<Vector3D> intersects = surfaces.at(i)->intersections(ray);
		for (int j = 0; j < intersects.size(); j++)
		{
			Vector3D intersection = intersects.at(j);
			Surface* surface = surfaces.at(i);
			returnVector.push_back(
				RaySurfaceIntersection(
					intersection,
					surface->getNormal(intersection),
					surface->getColour(),
					surface->getSpecularCoefficient()
				)
			);
		}
	}

	return returnVector;
}

void Object::addSurface(Surface* surface)
{
	surfaces.push_back(surface);
}

void Object::transform(Vector3D translation, Vector3D rotation)
{
	for (int i = 0; i < surfaces.size(); i++)
	{
		surfaces.at(i)->transform(translation, rotation);
	}
}

void Object::translate(Vector3D translation)
{
	for (int i = 0; i < surfaces.size(); i++)
	{
		surfaces.at(i)->translate(translation);
	}
}

void Object::rotate(Vector3D rotation)
{
	for (int i = 0; i < surfaces.size(); i++)
	{
		surfaces.at(i)->rotate(rotation);
	}
}

void Object::setColour(Rgb colour)
{
	this->colour = colour;
}

Rgb Object::getColour()
{
	return this->colour;
}