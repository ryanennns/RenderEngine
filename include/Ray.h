#pragma once
#include "./Vector3D.h"

class Ray
{
private:
	Vector3D A;
	Vector3D B;
public:
	Ray();
	Ray(Vector3D, Vector3D);
	Ray(Vector3D*, Vector3D*);

	Vector3D getA();
	Vector3D getOrigin();
	Vector3D getB();
	Vector3D getDirection();

	Vector3D evaluate(double);
	Ray transform(Vector3D);
};