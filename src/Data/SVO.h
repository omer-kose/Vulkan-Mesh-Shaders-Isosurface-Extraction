#pragma once

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <cstddef>
#include <memory>
#include <map>
#include <functional>
#include <array>

// SVONodeGPU used by GPU upload and tests (level included for diagnostics)
struct SVONodeGPU
{
    glm::vec3 lowerCorner;
    glm::vec3 upperCorner;
    uint8_t   colorIndex;
    int32_t   level;      // level: 0 = finest voxels, >0 = coarser
};

// Custom comparator for glm::uvec3
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
    std::vector<uint32_t> selectNodes(const glm::vec3& cameraPos, float lodBaseDist) const;
    size_t estimateMemoryUsageBytes() const;

private:
    // Compact node representation with explicit children indices
    struct Node
    {
        uint32_t childrenMask = 0;            // which children exist (bit 0..7)
        uint8_t  color = 0;                   // palette index (0==empty)
        int32_t  parentIndex = -1;            // index of parent node, -1 = root candidate
        std::array<int32_t, 8> children;      // explicit child indices, -1 if absent
        glm::uvec3 coord;                     // coordinates in padded index space for this level
        int       level = 0;                  // 0 = finest (voxels)
        int32_t   flatIndex = -1;             // index into flatNodesGPU after flatten

        Node(int l = 0, const glm::uvec3& c = glm::uvec3(0), uint8_t col = 0)
            : children{ -1,-1,-1,-1,-1,-1,-1,-1 }, coord(c), level(l), color(col), flatIndex(-1)
        {
        }
    };

    const std::vector<uint8_t>& origGrid;
    glm::uvec3 originalGridSize;
    glm::uvec3 paddedGridSize;
    glm::vec3  worldLower;
    glm::vec3  worldUpper;
    glm::vec3  voxelSize;
    int levels;

    std::vector<Node> nodes;           // All nodes in compact format
    std::vector<SVONodeGPU> flatNodesGPU;

    inline uint8_t voxelValue(const glm::uvec3& idx) const;
    static inline uint32_t gridLinear(const glm::uvec3& i, const glm::uvec3& s);
    void buildTree();
    void flattenTree();
    void computeWorldAABB(const Node& node, glm::vec3& outMin, glm::vec3& outMax) const;
};