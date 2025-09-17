#include "SVO.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <cassert>

static inline uint32_t nextPow2(uint32_t v)
{
    if(v == 0) return 1u;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

inline uint32_t SVO::gridLinear(const glm::uvec3& i, const glm::uvec3& s)
{
    return i.x + s.x * (i.y + s.y * i.z);
}

inline uint8_t SVO::voxelValue(const glm::uvec3& idx) const
{
    if(idx.x >= originalGridSize.x || idx.y >= originalGridSize.y || idx.z >= originalGridSize.z)
        return 0u;
    uint32_t linear = idx.x + originalGridSize.x * (idx.y + originalGridSize.y * idx.z);
    return origGrid[linear];
}

SVO::SVO(const std::vector<uint8_t>& grid,
    const glm::uvec3& originalGridSize_,
    const glm::vec3& worldLower_,
    const glm::vec3& worldUpper_)
    : 
    origGrid(grid),
    originalGridSize(originalGridSize_),
    worldLower(worldLower_),
    worldUpper(worldUpper_)
{
    uint32_t maxDim = std::max({ originalGridSize.x, originalGridSize.y, originalGridSize.z });
    uint32_t cubeDim = nextPow2(maxDim);
    paddedGridSize = glm::uvec3(cubeDim);

    // voxelSize measured in padded voxels -> world tiling aligns with padded coords
    voxelSize = (worldUpper - worldLower) / glm::vec3(paddedGridSize);

    // compute levels
    levels = 0;
    uint32_t tmp = cubeDim;
    while(tmp > 1u) { tmp >>= 1u; ++levels; }
    ++levels; // levels is such that level index runs 0..levels-1 and top has size cubeDim/(1<<top) == 1

    // compute brick leaf level (level index where a node covers BRICK_SIZE^3 padded voxels)
    // level==0 -> node covers 1 voxel. So leafLevel = log2(BRICK_SIZE). Voxel level processing will be done in the task shader
    int blk = BRICK_SIZE;
    leafLevel = 0;
    while((1 << leafLevel) < blk) ++leafLevel;
    assert((1 << leafLevel) == BRICK_SIZE && "BRICK_SIZE must be a power of two");
    // sanity: leafLevel must be <= levels-1
    if(leafLevel > levels - 1) leafLevel = levels - 1;

    buildTree();
    flattenTree();
}

void SVO::buildTree()
{
    using NodeMap = std::map<glm::uvec3, uint32_t, UVec3Comparator>;
    std::vector<NodeMap> levelMaps(levels);

    // Reserve some memory heuristically
    nodes.reserve(1024);
    bricks.reserve(1024);

    // --- Create brick nodes at leafLevel (sweep bricks over padded domain) ---
    glm::uvec3 padded = paddedGridSize;
    glm::uvec3 brickGrid = glm::uvec3(padded.x / BRICK_SIZE, padded.y / BRICK_SIZE, padded.z / BRICK_SIZE);

    for(uint32_t bz = 0; bz < brickGrid.z; ++bz)
        for(uint32_t by = 0; by < brickGrid.y; ++by)
            for(uint32_t bx = 0; bx < brickGrid.x; ++bx)
            {
                glm::uvec3 brickCoord(bx, by, bz);     // coord at level = leafLevel
                glm::uvec3 baseVoxel = brickCoord * glm::uvec3(BRICK_SIZE); // top-left-back voxel (padded-space)

                Brick b;
                bool anyNonZero = false;
                bool mono = true; // Mono is true if whole brick in non-empty and a singular color. 
                uint8_t monoColor = 0;
                bool firstFound = false;

                // fill brick
                for(int zz = 0; zz < BRICK_SIZE; ++zz)
                    for(int yy = 0; yy < BRICK_SIZE; ++yy)
                        for(int xx = 0; xx < BRICK_SIZE; ++xx)
                        {
                            glm::uvec3 v = baseVoxel + glm::uvec3((uint32_t)xx, (uint32_t)yy, (uint32_t)zz);
                            uint8_t val = voxelValue(v);
                            if(val != 0) 
                            {
                                b.voxels[xx + BRICK_SIZE * (yy + BRICK_SIZE * zz)] = val;
                                anyNonZero = true;
                                if(!firstFound) { monoColor = val; firstFound = true; }
                                else if(mono && val != monoColor) mono = false;
                            }
                            else
                            {
                                // If there is at least one empty voxel in the brick, the brick cannot be represented by a single color and node. It must be fully stored
                                mono = false;
                            }
                        }

                if(!anyNonZero) 
                {
                    continue; // skip empty bricks
                }

                // If the brick is mono-color and fully covered (no empty voxels), we will not store the brick data; instead create a mono-color node
                if(mono) 
                {
                    nodes.emplace_back(leafLevel, brickCoord, monoColor);
                    uint32_t nodeIdx = static_cast<uint32_t>(nodes.size() - 1);
                    nodes[nodeIdx].brickIndex = -1; // no brick store (mono color)
                    levelMaps[leafLevel][brickCoord] = nodeIdx;
                }
                else 
                {
                    // store the brick and create a node referencing it
                    bricks.push_back(b);
                    nodes.emplace_back(leafLevel, brickCoord, 0u);
                    uint32_t nodeIdx = static_cast<uint32_t>(nodes.size() - 1);
                    nodes[nodeIdx].brickIndex = static_cast<int32_t>(bricks.size() - 1);
                    levelMaps[leafLevel][brickCoord] = nodeIdx;
                    // Leaves will be processed in voxel level so the value stored in the brick is important. However, for each leaf, a representative node color must be selected so that it can propogate to the parents
                    std::array<int, 256> counts; counts.fill(0);
                    for(auto c : b.voxels) if(c) counts[c]++;
                    int best = 0; uint8_t bestColor = 0;
                    for(int c = 0; c < 256; ++c) if(counts[c] > best) { best = counts[c]; bestColor = static_cast<uint8_t>(c); }
                    nodes[nodeIdx].color = bestColor;
                }
            }

    // --- Build upper levels sparsely from level = leafLevel+1 .. levels-1 ---
    for(int L = leafLevel + 1; L < levels; ++L) 
    {
        // snapshot child entries to avoid invalidation issues
        std::vector<std::pair<glm::uvec3, uint32_t>> childEntries;
        childEntries.reserve(levelMaps[L - 1].size());
        for(const auto& e : levelMaps[L - 1]) childEntries.push_back(e);

        // collect children per parent coordinate
        std::map<glm::uvec3, std::vector<uint32_t>, UVec3Comparator> parentChildren;
        for(const auto& entry : childEntries) 
        {
            glm::uvec3 childCoord = entry.first;      // coord at level L-1
            uint32_t childIdx = entry.second;
            glm::uvec3 parentCoord = childCoord >> 1u; // coord at level L
            parentChildren[parentCoord].push_back(childIdx);
        }

        // create parents
        for(auto& pc : parentChildren) 
        {
            glm::uvec3 parentCoord = pc.first;
            const std::vector<uint32_t>& childrenIdx = pc.second;

            nodes.emplace_back(L, parentCoord, 0u);
            uint32_t parentIdx = static_cast<uint32_t>(nodes.size() - 1);
            levelMaps[L][parentCoord] = parentIdx;
            Node& parent = nodes[parentIdx];

            // link explicit children
            for(uint32_t childIdx : childrenIdx) 
            {
                Node& child = nodes[childIdx];
                // childCoord at level L-1:
                glm::uvec3 childCoord = glm::uvec3(child.coord);
                glm::uvec3 childOffset = childCoord & glm::uvec3(1u); // low bits
                uint32_t childSlot = childOffset.x | (childOffset.y << 1u) | (childOffset.z << 2u);
                parent.children[childSlot] = static_cast<int32_t>(childIdx);
                parent.childrenMask |= (1u << childSlot);
                child.parentIndex = static_cast<int32_t>(parentIdx);
            }


            // compute majority color fallback 
            std::array<int, 256> counts; counts.fill(0);
            for(int i = 0; i < 8; ++i) 
            {
                int32_t cidx = parent.children[i];
                if(cidx >= 0) counts[nodes[cidx].color]++;
            }
            int best = 0; uint8_t bestColor = 0;
            for(int c = 0; c < 256; ++c) if(counts[c] > best) { best = counts[c]; bestColor = static_cast<uint8_t>(c); }
            parent.color = bestColor;
            
        }
    }

    // done building nodes (sparse). nodes[] contains leaf bricks (or mono-color bricks) and parents up to root.
}

void SVO::flattenTree()
{
    flatNodesGPU.clear();
    flatNodesGPU.reserve(nodes.size());

    // simple DFS from root candidates (parentIndex == -1)
    std::vector<int32_t> roots;
    roots.reserve(32);
    for(size_t i = 0; i < nodes.size(); ++i) if(nodes[i].parentIndex == -1) roots.push_back((int32_t)i);

    std::function<void(int32_t)> dfs = [&](int32_t idx) {
        if(idx < 0) return;
        Node& n = nodes[idx];

        glm::vec3 mn, mx;
        computeWorldAABB(n, mn, mx);

        SVONodeGPU gn;
        gn.lowerCorner = mn;
        gn.upperCorner = mx;
        gn.colorIndex = n.color;
        gn.level = n.level;
        gn.brickIndex = (n.brickIndex >= 0) ? static_cast<uint32_t>(n.brickIndex) : UINT32_MAX;

        n.flatIndex = static_cast<int32_t>(flatNodesGPU.size());
        flatNodesGPU.push_back(gn);

        // traverse children in canonical order
        for(int i = 0; i < 8; ++i) {
            int32_t c = n.children[i];
            if(c >= 0) dfs(c);
        }
        };

    for(int32_t r : roots) dfs(r);
}

void SVO::computeWorldAABB(const Node& node, glm::vec3& outMin, glm::vec3& outMax) const
{
    float scale = static_cast<float>(1u << node.level); // how many padded voxels per node side
    glm::vec3 nodeSize = voxelSize * scale;
    outMin = worldLower + glm::vec3(node.coord) * nodeSize;
    outMax = outMin + nodeSize;
    outMin = glm::max(outMin, worldLower);
    outMax = glm::min(outMax, worldUpper);
}

std::vector<uint32_t> SVO::selectNodes(const glm::vec3& cameraPos, float lodBaseDist) const
{
    std::vector<uint32_t> result;
    result.reserve(512);

    // iterative stack traversal starting at roots
    std::vector<int32_t> stack;
    for(size_t i = 0; i < nodes.size(); ++i) 
    {
        if(nodes[i].parentIndex == -1) stack.push_back((int32_t)i);
    }

    while(!stack.empty()) 
    {
        int32_t nodeIdx = stack.back();
        stack.pop_back();
        const Node& n = nodes[nodeIdx];

        glm::vec3 mn, mx;
        // compute aabb
        float scale = static_cast<float>(1u << n.level);
        glm::vec3 nodeSize = voxelSize * scale;
        mn = worldLower + glm::vec3(n.coord) * nodeSize;
        mx = mn + nodeSize;
        mn = glm::max(mn, worldLower);
        mx = glm::min(mx, worldUpper);

        glm::vec3 center = (mn + mx) * 0.5f;
        float dist = glm::length(cameraPos - center);
        glm::vec3 ext = mx - mn;
        float nodeExtent = std::max(std::max(ext.x, ext.y), ext.z);

        // Determine if node is a leaf in our bricked SVO
        bool isLeaf = (n.childrenMask == 0) || (n.brickIndex >= 0);

        // Select leaf nodes unconditionally (they contain renderable data), or select
        // non-leaf nodes when they are sufficiently far (coarser LOD).
        if(isLeaf || dist > lodBaseDist * nodeExtent) 
        {
            if(n.flatIndex >= 0) 
            {
                result.push_back(static_cast<uint32_t>(n.flatIndex));
            }
        }
        else 
        {
            // descend children (only existing ones)
            for(int i = 0; i < 8; ++i) 
            {
                int32_t c = n.children[i];
                if(c >= 0) stack.push_back(c);
            }
        }
    }

    return result;
}

std::vector<uint32_t> SVO::selectNodesScreenSpace(const glm::vec3& cameraPos, float fovY, float aspect, uint32_t screenHeight, float pixelThreshold) const
{
    // Pixels-per-world-unit factor at distance = 1
    const float screenFactor = float(screenHeight) / (2.0f * glm::tan(fovY * 0.5f));

    std::vector<uint32_t> result;
    result.reserve(1024);
    std::vector<int32_t> stack;

    // Push root nodes
    for(size_t i = 0; i < nodes.size(); ++i) 
    {
        if(nodes[i].parentIndex == -1) stack.push_back((int32_t)i);
    }

    while(!stack.empty()) 
    {
        int32_t nodeIdx = stack.back();
        stack.pop_back();
        const Node& n = nodes[nodeIdx];

        glm::vec3 min, max;
        computeWorldAABB(n, min, max);

        glm::vec3 center = (min + max) * 0.5f;
        float dist = glm::length(cameraPos - center);
        glm::vec3 ext = max - min;
        float nodeExtent = std::max(std::max(ext.x, ext.y), ext.z);

        // Calculate the screen-space size of this node
        if(dist <= 0.0f) dist = 0.001f; // Avoid division by zero
        float screenSize = (nodeExtent * screenFactor) / dist;

        // Check if the node's screen size is larger than our error threshold.
        // If it is, we need more detail (descend further).
        bool needsRefinement = (screenSize > pixelThreshold);

        bool isLeaf = (n.childrenMask == 0) || (n.brickIndex >= 0);

        // Select this node if:
        // 1. It's a leaf (we can't subdivide further), OR
        // 2. Its projected size is small enough that we don't need to refine it.
        if(isLeaf || !needsRefinement) 
        {
            if(n.flatIndex >= 0) 
            {
                result.push_back(static_cast<uint32_t>(n.flatIndex));
            }
        }
        else 
        {
            // The node is too big on screen and is not a leaf -> descend.
            for(int i = 0; i < 8; ++i) 
            {
                int32_t c = n.children[i];
                if(c >= 0) stack.push_back(c);
            }
        }
    }

    return result;
}

size_t SVO::estimateMemoryUsageBytes() const
{
    size_t memory = 0;
    memory += nodes.size() * sizeof(Node);
    memory += flatNodesGPU.size() * sizeof(SVONodeGPU);
    memory += bricks.size() * sizeof(Brick);
    return memory;
}

uint32_t SVO::getLeafLevel() const
{
    return leafLevel;
}
