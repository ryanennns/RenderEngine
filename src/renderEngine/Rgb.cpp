#include "../include/Rgb.h"
#include <algorithm>
#include "stdio.h"

Rgb::Rgb()
{
	this->r = (unsigned char)0.0;
	this->g = (unsigned char)0.0;
	this->b = (unsigned char)0.0;
}

Rgb::Rgb(unsigned char r, unsigned char g, unsigned char b)
{
	this->r = r;
	this->g = g;
	this->b = b;
}

unsigned char Rgb::getRed()
{
	return this->r;
}

unsigned char Rgb::getGreen()
{
	return this->g;
};

unsigned char Rgb::getBlue()
{
	return this->b;
}

void Rgb::setRed(unsigned char value)
{
	this->r = value;
}

void Rgb::setGreen(unsigned char value)
{
	this->g = value;
};

void Rgb::setBlue(unsigned char value)
{
	this->b = value;
}

Rgb Rgb::average(Rgb other)
{
	return Rgb(
		(unsigned char)((this->r + other.r) / 2),
		(unsigned char)((this->g + other.g) / 2),
		(unsigned char)((this->b + other.b) / 2)
	);
}

Rgb Rgb::operator*(double scalar)
{
	if (this->r * scalar > 255 ||
		this->g * scalar > 255 ||
		this->b * scalar > 255)
	{
		printf("error: scalar value is too high\n");
		printf("r: %lf, g: %lf, b: %lf\n", this->r * scalar, this->g * scalar, this->b * scalar);
	}

	return Rgb(
		(unsigned char)(this->r * scalar),
		(unsigned char)(this->g * scalar),
		(unsigned char)(this->b * scalar)
	);
}

Rgb Rgb::operator*(Rgb other)
{
	if (
		(this->r * other.r) > 255 ||
		(this->g * other.g) > 255 ||
		(this->b * other.b) > 255
		)
	{
		printf("error: scalar value is too high\n");
		printf("r: %d, g: %d, b: %d\n", this->r * other.r, this->g * other.g, this->b * other.b);
	}

	return Rgb(
		(unsigned char)(this->r * other.r), // , 255.0),
		(unsigned char)(this->g * other.g), // , 255.0),
		(unsigned char)(this->b * other.b) // , 255.0)
	);
}

Rgb Rgb::operator+(Rgb other)
{
	return Rgb(
		(unsigned char)std::min((this->r + other.r), 255),
		(unsigned char)std::min((this->g + other.g), 255),
		(unsigned char)std::min((this->b + other.b), 255)
	);
}

Rgb Rgb::operator-(Rgb other)
{
	return Rgb(
		(unsigned char)std::min((this->r - other.r), 0),
		(unsigned char)std::min((this->g - other.g), 0),
		(unsigned char)std::min((this->b - other.b), 0)
	);
}