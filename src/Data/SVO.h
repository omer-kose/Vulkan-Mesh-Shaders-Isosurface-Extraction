#pragma once

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <cstddef>
#include <memory>
#include <map>
#include <functional>
#include <array>

/*
    Normally, in an SVO, I would let user to choose the brick size. However, this SVO's bricks are processed by the task shader. Therefore, it's size cannot be arbitrary due to limits.

    4 is the best choice as it creates a volume of 64 voxels to process. Task Shaders work in groups of 32 threads preferred by the spec and with size 64 this means 2 iterations per thread which is 
    good amount of work per GPU thread. Also, it keeps TaskPayload in reasonable sizes. 

    Tried 2 and 8 too, they perform much worse. 
    2: Too many leaves and thus too many task shader dispatches which is an overhead as an overhead for the driver and causes performance drop
    8: Too many work per GPU thread (16 iterations)
*/
constexpr int FINE_BRICK_SIZE = 4;
constexpr int FINE_BRICK_VOLUME = FINE_BRICK_SIZE * FINE_BRICK_SIZE * FINE_BRICK_SIZE;
constexpr int COARSE_BRICK_SIZE = 2;
constexpr int COARSE_BRICK_VOLUME = COARSE_BRICK_SIZE * COARSE_BRICK_SIZE * COARSE_BRICK_SIZE;

struct SVONodeGPU
{
    glm::vec3 lowerCorner;
    glm::vec3 upperCorner;
    uint8_t   colorIndex;
    uint8_t   level;      // 0 = finest voxels (we use bricks at leafLevel)
    uint32_t  brickIndex; // UINT32_MAX => no brick present (mono-color leaf or internal)
};

/*
    Simple fixed-size brick types (stores BRICK_SIZE ^ 3 bytes)
    There are two possible options:
    LOD0 -> 4x4x4 fine bricks 
    LOD1 -> 2x2x2 coarse bricks
    LOD2 and on -> collapsed into nodes

    As, these bricks will be used in shaders, they have to be fixed size structs. Therefore, creating two variations of it. Technically, you would need log2(FINE_BRICK_SIZE) different types but as 
    FINE_BRICK_SIZE is always 4, I can just get away by having 2 structs. Not the proudest solution I have found.
*/
struct FineBrick
{
    std::array<uint8_t, FINE_BRICK_VOLUME> voxels;
};

struct CoarseBrick
{
    std::array<uint8_t, COARSE_BRICK_VOLUME> voxels;
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

    const std::vector<SVONodeGPU>& getFlatGPUNodes();
    const std::vector<FineBrick>& getFineBricks();
    const std::vector<CoarseBrick>& getCoarseBricks();
    void clearBricks();
    std::vector<uint32_t> selectNodes(const glm::vec3& cameraPos, float lodBaseDist) const;
    std::vector<uint32_t> selectNodesScreenSpace(const glm::vec3& cameraPos, float fovY, float aspect, uint32_t screenHeight, float pixelThreshold) const;
    size_t estimateMemoryUsageBytes() const;
    uint32_t getLeafLevel() const;

    // For unit tests
    friend class SVOUnitTests;
private:
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

        Node(int l = 0, const glm::uvec3& c = glm::uvec3(0), uint8_t col = 0)
            : coord(glm::u16vec3(c)), parentIndex(-1), children{ -1,-1,-1,-1,-1,-1,-1,-1 },
            childrenMask(0), flatIndex(-1), brickIndex(-1), level(static_cast<uint8_t>(l)), color(col)
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
    int leafLevel;        // level at which bricks live (log2(FINE_BRICK_SIZE))

    std::vector<Node> nodes;                 // sparse nodes (only non-empty / bricks)
    std::vector<SVONodeGPU> flatNodesGPU;    // flattened snapshot for GPU
    std::vector<FineBrick> fineBricks;               // dense brick data when necessary
    std::vector<CoarseBrick> fineBrickMips; // first mip level of fine bricks temporarily used for creating coarse bricks
    std::vector<CoarseBrick> coarseBricks;

    inline uint8_t voxelValue(const glm::uvec3& idx) const;
    static inline uint32_t gridLinear(const glm::uvec3& i, const glm::uvec3& s);
    void buildTree();
    void flattenTree();
    void computeWorldAABB(const Node& node, glm::vec3& outMin, glm::vec3& outMax) const;
    CoarseBrick computeFineBrickMip(const FineBrick& b);
};
