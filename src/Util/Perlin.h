#pragma once
#include <vector>
#include <random>

// Simple classic Perlin noise (3D) implementation with permutation table
class Perlin
{
public:
    Perlin(uint32_t seed = 1337)
    {
        init(seed);
    }

    void init(uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        perm.resize(512);
        // build permutation 0..255 and shuffle
        std::vector<int> p(256);
        for(int i = 0; i < 256; ++i) p[i] = i;
        for(int i = 255; i > 0; --i) std::swap(p[i], p[rng() % (i + 1)]);
        for(int i = 0; i < 512; ++i) perm[i] = p[i & 255];
    }

    // 3D Perlin noise in range [-1,1]
    float noise(float x, float y, float z) const
    {
        // Find unit cube
        int X = fastfloor(x) & 255;
        int Y = fastfloor(y) & 255;
        int Z = fastfloor(z) & 255;

        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);

        float u = fade(x);
        float v = fade(y);
        float w = fade(z);

        int A = perm[X] + Y;
        int AA = perm[A] + Z;
        int AB = perm[A + 1] + Z;
        int B = perm[X + 1] + Y;
        int BA = perm[B] + Z;
        int BB = perm[B + 1] + Z;

        float res = lerp(w,
            lerp(v,
                lerp(u, grad(perm[AA], x, y, z),
                    grad(perm[BA], x - 1, y, z)),
                lerp(u, grad(perm[AB], x, y - 1, z),
                    grad(perm[BB], x - 1, y - 1, z))),
            lerp(v,
                lerp(u, grad(perm[AA + 1], x, y, z - 1),
                    grad(perm[BA + 1], x - 1, y, z - 1)),
                lerp(u, grad(perm[AB + 1], x, y - 1, z - 1),
                    grad(perm[BB + 1], x - 1, y - 1, z - 1))));
        // res is roughly in [-1,1]
        return res;
    }

private:
    std::vector<int> perm;
    static inline int fastfloor(float x) { return (x > 0) ? (int)x : ((int)x - 1); }
    static inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    static inline float lerp(float t, float a, float b) { return a + t * (b - a); }
    static inline float grad(int hash, float x, float y, float z)
    {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    }
};

// Fractal Brownian Motion (FBM) helper
inline float fbm3D(const Perlin& perlin, float x, float y, float z, int octaves,
    float lacunarity, float gain)
{
    float freq = 1.0f;
    float amp = 1.0f;
    float sum = 0.0f;
    float norm = 0.0f;
    for(int i = 0; i < octaves; ++i) {
        sum += amp * perlin.noise(x * freq, y * freq, z * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return sum / norm; // roughly in [-1,1]
}

inline float fbm2D(const Perlin& perlin, float x, float y, int octaves,
    float lacunarity, float gain)
{
    // call 3D with z=0 for simplicity
    return fbm3D(perlin, x, y, 0.0f, octaves, lacunarity, gain);
}