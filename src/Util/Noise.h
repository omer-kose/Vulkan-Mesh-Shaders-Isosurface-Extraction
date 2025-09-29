#pragma once
#include <cmath>

inline float hash3D(int x, int y, int z)
{
    int h = x * 374761393 + y * 668265263 + z * 73856093;
    h = (h ^ (h >> 13)) * 1274126177;
    return ((h & 0x7fffffff) / float(0x7fffffff));
}

inline float noise3D(float x, float y, float z, float scale = 0.05f)
{
    int xi = int(std::floor(x * scale));
    int yi = int(std::floor(y * scale));
    int zi = int(std::floor(z * scale));
    return hash3D(xi, yi, zi);
}