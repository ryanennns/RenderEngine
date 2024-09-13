#include "../../include/Ray.h"

Ray::Ray()
{
	this->A = Vector3D(0,0,0);
	this->B = Vector3D(1,0,0);
}

Ray::Ray(Vector3D origin, Vector3D direction)
{
	this->A = origin;
	this->B = direction;
}

Ray::Ray(Vector3D* origin, Vector3D* direction)
{
	this->A = Vector3D(origin);
	this->B = Vector3D(direction);
}

Vector3D Ray::getA()
{
	return this->A;
}

Vector3D Ray::getOrigin()
{
	return this->A;
}
	
Vector3D Ray::getB()
{
	return this->B;
}

Vector3D Ray::getDirection()
{
	return this->B.subtract(this->A);
}

Vector3D Ray::evaluate(double t)
{
	return this->A.add(this->getDirection().multiply(t));
}

Ray Ray::transform(Vector3D v)
{
	return Ray(this->A.add(v), this->B.add(v));
}