#pragma once

#include "Object.h"
#include "Light.h"
#include "HitDetection.h"
#include <vector>

class Scene
{
public:
	Scene();
	~Scene();

	void addObject(Object*);
	void addLight(Light*);
	std::vector<Object*> getObjects();
	std::vector<Light*> getLights();
	std::vector<HitDetection> intersections(Ray);
private:
	std::vector<Object*> objects;
	std::vector<Light*> lights;
};