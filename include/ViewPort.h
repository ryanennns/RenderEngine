#pragma once

#include <vector>
#include "../include/Vector3D.h"
#include "../include/Ray.h"

using namespace std;

class ViewPort
{
private:
	Vector3D eye;
	vector<vector<Ray>> rays;
public:
	vector<vector<Ray>> generateRays(int width, int height) const;
	ViewPort();
	ViewPort(Vector3D eye, double width = 400, double height = 800);

	vector<vector<Ray>> getRays();
	Vector3D getEye();
};