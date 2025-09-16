#pragma once
#include <Util/Perlin.h>

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <fmt/core.h>

/// generateVoxelTerrain
/// - size: grid dimensions (x,y,z)
/// - worldLower/worldUpper: world bounds; grid maps uniformly to these
/// - baseSeed: RNG seed
/// - parameters: tuning for mountains, caves, clouds, terracing, etc
struct TerrainParams
{
    uint32_t seed = 1337;
    // heightfield
    int heightOctaves = 6;
    float heightFrequency = 1.0f / 128.0f; // adjust relative to grid
    float heightLacunarity = 2.0f;
    float heightGain = 0.5f;
    float heightAmplitude = 200.0f; // how tall mountains are in voxel units

    // terracing
    bool enableTerrace = false;
    float terraceStep = 4.0f; // voxel steps

    // caves (3D subtractive noise)
    bool enableCaves = true;
    int caveOctaves = 3;
    float caveFrequency = 1.0f / 32.0f;
    float caveThreshold = 0.45f; // higher => fewer caves

    // clouds
    bool enableClouds = true;
    int cloudBlockStride = 64; // position grid spacing for cloud slabs
    int cloudThickness = 16;
    uint8_t cloudColor = 200; // palette index for clouds

    // bedrock / surface materials
    uint8_t bedrockColor = 3;
    uint8_t stoneColor = 4;
    uint8_t grassColor = 5;
    uint8_t dirtColor = 6;

    // sea level (if you want oceans)
    int seaLevel = 32;

    // downscale speedups: skip inner loops for very large grids
    bool fastMode = false; // if true, stride=1 normally, else you may run with small stride
};

struct VoxelColor
{
    uint8_t color[4]; // also compatible with vox files
};

static inline uint8_t clamp8(int v) { return (uint8_t)std::min(255, std::max(0, v)); }

std::vector<uint8_t> generateVoxelTerrain(
    const glm::uvec3& gridSize,
    const glm::vec3& worldLower,
    const glm::vec3& worldUpper,
    const TerrainParams& params)
{
    fmt::println("Generating voxel terrain {}x{}x{}, seed={}...", gridSize.x, gridSize.y, gridSize.z, params.seed);
    Perlin perlin(params.seed);

    size_t total = size_t(gridSize.x) * gridSize.y * gridSize.z;
    std::vector<uint8_t> grid;
    grid.assign(total, 0);

    // grid -> world scale
    glm::vec3 worldExtent = worldUpper - worldLower;
    auto voxelCenterWorld = [&](uint32_t x, uint32_t y, uint32_t z) -> glm::vec3 {
        glm::vec3 idx((float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f);
        return worldLower + (idx / glm::vec3(gridSize)) * worldExtent;
        };

    // Precompute freq scales (work in grid-space to keep nice)
    float hf = params.heightFrequency;
    float cf = params.caveFrequency;
    float caveThresh = params.caveThreshold;

    // loop y as height to accelerate (we compute height first)
    // compute 2D FBM heightfield in grid coords
    std::vector<float> heightMap;
    heightMap.resize(size_t(gridSize.x) * gridSize.z);
    for(uint32_t z = 0; z < gridSize.z; ++z) 
    {
        for(uint32_t x = 0; x < gridSize.x; ++x) 
        {
            float nx = (float)x * hf;
            float nz = (float)z * hf;
            float h = fbm2D(perlin, nx, nz, params.heightOctaves, params.heightLacunarity, params.heightGain);
            // h in [-1,1] -> scale to [0,1]
            h = (h * 0.5f) + 0.5f;
            // scale to amplitude
            float worldH = h * params.heightAmplitude + params.seaLevel;
            // map to voxel-space height
            // note: heights exceed gridSize.y will be clamped
            float vy = worldH;
            if(params.enableTerrace) {
                vy = std::floor(vy / params.terraceStep) * params.terraceStep;
            }
            // clamp to grid's y
            int vi = std::max(0, std::min((int)gridSize.y - 1, int(std::round(vy))));
            heightMap[x + z * gridSize.x] = (float)vi;
        }
    }

    // fill columns (x,z) up to heightMap with materials
    for(uint32_t z = 0; z < gridSize.z; ++z) 
    {
        for(uint32_t x = 0; x < gridSize.x; ++x) 
        {
            int maxY = (int)heightMap[x + z * gridSize.x];
            // bedrock bottom
            int bedrockThickness = 3;
            for(int y = 0; y <= maxY && y < (int)gridSize.y; ++y) {
                size_t idx = x + gridSize.x * (y + gridSize.y * z);
                if(y < bedrockThickness) {
                    grid[idx] = params.bedrockColor; // bedrock
                }
                else if(y == maxY) {
                    grid[idx] = params.grassColor; // top
                }
                else if(y > maxY - 4) {
                    grid[idx] = params.dirtColor; // near-surface
                }
                else {
                    grid[idx] = params.stoneColor; // stone
                }
            }
        }
    }

    // Caves: carve out with 3D FBM noise (subtract)
    if(params.enableCaves) 
    {
        fmt::println("  carving caves...");
        for(uint32_t z = 0; z < gridSize.z; ++z) 
        {
            for(uint32_t y = 0; y < gridSize.y; ++y) 
            {
                for(uint32_t x = 0; x < gridSize.x; ++x) 
                {
                    // only check where we have solid
                    size_t idx = x + gridSize.x * (y + gridSize.y * z);
                    if(grid[idx] == 0) continue;
                    // sample 3D fbm
                    float fx = x * cf;
                    float fy = y * cf;
                    float fz = z * cf;
                    float n = fbm3D(perlin, fx, fy, fz, params.caveOctaves, 2.0f, 0.5f);
                    // n in [-1,1] -> [0,1]
                    n = 0.5f + 0.5f * n;
                    if(n > caveThresh) 
                    {
                        // carve cave -> empty
                        grid[idx] = 0;
                    }
                }
            }
        }
    }

    // Clouds: place slabs of mono-color clouds in high altitudes
    if(params.enableClouds) 
    {
        fmt::println("  adding clouds...");
        for(uint32_t bx = 0; bx < gridSize.x; bx += params.cloudBlockStride) 
        {
            for(uint32_t bz = 0; bz < gridSize.z; bz += params.cloudBlockStride) 
            {
                // small random offset to pattern
                for(int dx = 0; dx < params.cloudBlockStride / 2 && bx + (uint32_t)dx < gridSize.x; ++dx) 
                {
                    for(int dz = 0; dz < params.cloudBlockStride / 2 && bz + (uint32_t)dz < gridSize.z; ++dz) 
                    {
                        // cloud slab area
                        uint32_t sx = bx + dx;
                        uint32_t sz = bz + dz;
                        // cloud height band in grid coordinates (top-of-world region)
                        uint32_t cy0 = std::min(gridSize.y - 1, gridSize.y - 100);
                        for(uint32_t cy = cy0; cy < std::min(gridSize.y, cy0 + (uint32_t)params.cloudThickness); ++cy) 
                        {
                            size_t idx = sx + gridSize.x * (cy + gridSize.y * sz);
                            grid[idx] = params.cloudColor;
                        }
                    }
                }
            }
        }
    }

    fmt::println("done.");
    return grid;
}

// Build a color lookup table for terrain materials.
// Index 0 = empty (transparent).
inline std::vector<VoxelColor> buildTerrainColorTable(const TerrainParams& params)
{
    // max index is 255 (uint8_t), but we only need a few entries
    std::vector<VoxelColor> table(256, { 0,0,0,0 });

    // Convention: [R,G,B,A] with A=255 = opaque
    auto set = [&](uint8_t idx, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        table[idx] = { r,g,b,a };
    };

    // Assign materials according to params
    set(params.bedrockColor, 40, 40, 40);   // dark gray
    set(params.stoneColor, 128, 128, 128);   // gray
    set(params.dirtColor, 120, 72, 48);   // brown
    set(params.grassColor, 34, 139, 34);   // forest green
    set(params.cloudColor, 220, 220, 220); // white

    return table;
}