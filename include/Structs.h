#pragma once

struct Coordinates
{
    double x, y;
};

struct Vector
{
    double x;
    double y;
    double z;
};

struct Line
{
    Vector a;
    Vector b;
};

struct Triangle
{
    Vector a;
    Vector b;
    Vector c;
};

struct Object
{
    int size;
    Triangle* triangles;
};

struct Landscape
{
    int size;
    Object* objects;
};

struct LineTriangleIntersection
{
    bool intersects;
    Triangle triangle;
    Line line;
    Vector intersection;
};
