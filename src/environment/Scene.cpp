#include <vector>
#include "../../include/Scene.h"
#include "../../include/HitDetection.h"
#include "../../include/Ray.h"
#include "../../include/RaySurfaceIntersection.h"

using namespace std;

Scene::Scene()
{

}

Scene::~Scene()
{
	for (int i = 0; i < this->objects.size(); i++)
	{
		//delete this->objects.at(i);
	}

	for (int i = 0; i < this->lights.size(); i++)
	{
		//delete this->lights.at(i);
	}
}

void Scene::addLight(Light* light)
{
	this->lights.push_back(light);
}

void Scene::addObject(Object* object)
{
	this->objects.push_back(object);
}

vector<Object*> Scene::getObjects()
{
	return this->objects;
}

vector<Light*> Scene::getLights()
{
	return this->lights;
}

vector<HitDetection> Scene::intersections(Ray ray)
{
	vector<HitDetection> intersections;
	for (int i = 0; i < objects.size(); i++)
	{
		vector<RaySurfaceIntersection> objectIntersections = objects.at(i)->intersections(ray);

		// VERY minor optimization to be had by only returning the first intersection
		for (int j = 0; j < objectIntersections.size(); j++)
		{
			RaySurfaceIntersection intersection = objectIntersections.at(0);
			intersections.push_back(
				HitDetection(
					intersection.getHitPoint(),
					intersection.getNormal(),
					intersection.getColour(),
					intersection.getSpecularCoefficient()
				)
			);
		}
	}
	return intersections;
}
