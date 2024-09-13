#pragma once

#include <vector>

#include "Scene.h"
#include "ViewPort.h"
#include "Rgb.h"
#include "HitDetection.h"

class ViewDriver
{
public:
	ViewDriver(Scene*, ViewPort);
	vector<vector<Rgb>> processFrame();
	bool isInShadow(HitDetection, Light*);
private:
	Rgb processLighting(HitDetection);
	Scene* scene;
	ViewPort viewPort;

	Vector3D reflect(Vector3D, Vector3D);
	Rgb getSpecular(
		Vector3D& lightDirection,
		Vector3D& normal,
		Vector3D& viewDirection,
		Rgb& objectColor,
		double specularCoefficient
	);
	Rgb getAmbient();
	Rgb getDiffuse(Vector3D& normal, Vector3D& lightDirection, Rgb& objectColor);
};