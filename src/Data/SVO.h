#pragma once

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <cstddef>
#include <memory>
#include <map>
#include <functional>
#include <array>

// Tune this according to memory vs quality tradeoff.
// Must be power of two. 8 is a good start for terrain.
constexpr int BRICK_SIZE = 8;
constexpr int BRICK_VOLUME = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;

struct SVONodeGPU
{
    glm::vec3 lowerCorner;
    glm::vec3 upperCorner;
    uint8_t   colorIndex;
    uint8_t   level;      // 0 = finest voxels (we use bricks at leafLevel)
    uint32_t  brickIndex; // UINT32_MAX => no brick present (mono-color leaf or internal)
};

// simple fixed-size brick type (stores BRICK_SIZE^3 bytes)
struct Brick
{
    std::array<uint8_t, BRICK_VOLUME> voxels;
};

struct UVec3Comparator
{
    bool operator()(const glm::uvec3& a, const glm::uvec3& b) const
    {
        if(a.x != b.x) return a.x < b.x;
        if(a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    }
};

class SVO
{
public:
    SVO(const std::vector<uint8_t>& grid,
        const glm::uvec3& originalGridSize,
        const glm::vec3& worldLower,
        const glm::vec3& worldUpper);

    const std::vector<SVONodeGPU>& getFlatGPUNodes() const { return flatNodesGPU; }
    const std::vector<Brick>& getBricks() const { return bricks; }
    std::vector<uint32_t> selectNodes(const glm::vec3& cameraPos, float lodBaseDist) const;
    std::vector<uint32_t> selecNodesScreenSpace(const glm::vec3& cameraPos, float fovY, float aspect, uint32_t screenHeight, float pixelThreshold) const;
    size_t estimateMemoryUsageBytes() const;
public:
    // Compact node representation with explicit children indices
    struct Node
    {
        glm::u16vec3 coord;                    // coordinate in padded grid for this level
        int32_t parentIndex = -1;              // parent index or -1
        std::array<int32_t, 8> children;       // child indices, -1 if absent
        uint8_t childrenMask = 0;              // bitmask of which children exist
        int32_t flatIndex = -1;                // index into flatNodesGPU
        int32_t brickIndex = -1;               // index into bricks (or -1 for none)
        uint8_t level = 0;                     // level (0=voxels, leafLevel=brick size)
        uint8_t color = 0;                     // aggregated color (mono color or majority)

        bool alive = true; // logical alive flag (later used for compaction of the tree)

        Node(int l = 0, const glm::uvec3& c = glm::uvec3(0), uint8_t col = 0)
            : coord(glm::u16vec3(c)), parentIndex(-1), children{ -1,-1,-1,-1,-1,-1,-1,-1 },
            childrenMask(0), flatIndex(-1), brickIndex(-1), level(static_cast<uint8_t>(l)), color(col), alive(true)
        {
        }
    };

    const std::vector<uint8_t>& origGrid;
    glm::uvec3 originalGridSize;
    glm::uvec3 paddedGridSize;
    glm::vec3  worldLower;
    glm::vec3  worldUpper;
    glm::vec3  voxelSize; // computed using paddedGridSize
    int levels;           // number of levels (level indices 0..levels-1)
    int leafLevel;        // level at which bricks live (log2(BRICK_SIZE))

    std::vector<Node> nodes;                 // sparse nodes (only non-empty / bricks)
    std::vector<SVONodeGPU> flatNodesGPU;    // flattened snapshot for GPU
    std::vector<Brick> bricks;               // dense brick data when necessary

    inline uint8_t voxelValue(const glm::uvec3& idx) const;
    static inline uint32_t gridLinear(const glm::uvec3& i, const glm::uvec3& s);
    void buildTree();
    void flattenTree();
    void computeWorldAABB(const Node& node, glm::vec3& outMin, glm::vec3& outMax) const;
    float distanceToAABB(const glm::vec3& p, const glm::vec3& min, const glm::vec3& max) const;
};
