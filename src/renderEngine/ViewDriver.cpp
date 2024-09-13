#include <vector>
#include <algorithm>
#include <cmath>
#include "../include/ViewDriver.h"
#include "../include/Sphere.h"
#include "../include/Light.h"

ViewDriver::ViewDriver(Scene* scene, ViewPort viewPort)
{
	this->scene = scene;
	this->viewPort = viewPort;
}

vector<vector<Rgb>> ViewDriver::processFrame()
{
	vector<vector<Rgb>> rgbValues;
	vector<vector<Ray>> rays = viewPort.getRays();

	for (int y = 0; y < rays.size(); y++)
	{
		vector<Ray> row = rays.at(y);
		vector<Rgb> rgbColumn;
		for (int x = 0; x < row.size(); x++)
		{
			Ray ray = row.at(x);
			vector<HitDetection> intersections = scene->intersections(ray);

			if (intersections.size() > 0)
			{
				rgbColumn.push_back(this->processLighting(intersections.at(0)));
			}
			else
			{
				rgbColumn.push_back(Rgb(12, 12, 12));
			}
		}
		rgbValues.push_back(rgbColumn);
	}

	return rgbValues;
}

Rgb ViewDriver::getSpecular(Vector3D& lightDirection, Vector3D& normal, Vector3D& viewDirection, Rgb& objectColor, double specularCoefficient)
{
	Vector3D reflectDirection = this->reflect(lightDirection.negative(), normal).normalize();
	double spec = std::pow(std::max(viewDirection.dotProduct(reflectDirection), 0.0), specularCoefficient);
	return (Rgb(200, 200, 200) + objectColor * 0.01) * 0.7 * spec;
}

Rgb ViewDriver::getAmbient()
{
	return Rgb(0, 0, 0); // Placeholder for ambient light calculation
}

Rgb ViewDriver::getDiffuse(Vector3D& normal, Vector3D& lightDirection, Rgb& objectColor)
{
	double diff = std::fabs(normal.dotProduct(lightDirection));
	return objectColor * std::max(diff, 0.01);
}

Rgb ViewDriver::processLighting(HitDetection intersection)
{
	Vector3D normal = intersection.getNormal().normalize();
	Vector3D hitPoint = intersection.getHitPoint();
	Vector3D viewDirection = this->viewPort.getEye().subtract(hitPoint).normalize();
	Rgb objectColour = intersection.getColour();

	Rgb finalColour;
	vector<Light*> lights = this->scene->getLights();

	for (int i = 0; i < lights.size(); i++)
	{
		Light* light = this->scene->getLights().at(0);
		Vector3D lightDirection = light->getOrigin().subtract(hitPoint).normalize();

		Rgb specular = this->getSpecular(lightDirection, normal, viewDirection, objectColour, intersection.getSpecularCoefficient());
		Rgb ambient = this->getAmbient();
		Rgb diffuse = this->getDiffuse(normal, lightDirection, objectColour);

		Rgb combinedLight = ambient + diffuse;

		if (this->isInShadow(intersection, light))
		{
			return combinedLight * 0.1;
		}

		combinedLight = combinedLight + specular;

		finalColour = combinedLight;
	}

	return finalColour;
}

bool ViewDriver::isInShadow(HitDetection intersection, Light* light)
{
	Ray intersectionToLight = Ray(intersection.getHitPoint(), light->getOrigin());
	vector<HitDetection> sceneIntersections = this->scene->intersections(intersectionToLight);
	return sceneIntersections.size() > 0;
}

Vector3D ViewDriver::reflect(Vector3D incident, Vector3D normal)
{
	incident = incident.normalize();
	normal = normal.normalize();
	return incident.subtract(normal.multiply(2.0 * incident.dotProduct(normal)));
}