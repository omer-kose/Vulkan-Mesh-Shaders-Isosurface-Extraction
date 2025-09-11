#include "SVO.h"
#include <algorithm>
#include <cmath>
#include <numeric>

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
    : origGrid(grid),
    originalGridSize(originalGridSize_),
    worldLower(worldLower_),
    worldUpper(worldUpper_)
{
    uint32_t maxDim = std::max({ originalGridSize.x, originalGridSize.y, originalGridSize.z });
    uint32_t cubeDim = nextPow2(maxDim);
    paddedGridSize = glm::uvec3(cubeDim);
    // after paddedGridSize is computed:
    voxelSize = (worldUpper - worldLower) / glm::vec3(paddedGridSize); // important: use paddedGridSize


    levels = 0;
    uint32_t tmp = cubeDim;
    while(tmp > 1u) { tmp >>= 1u; ++levels; }
    ++levels;

    buildTree();
    flattenTree();
}

void SVO::buildTree()
{
    using NodeMap = std::map<glm::uvec3, uint32_t, UVec3Comparator>;
    std::vector<NodeMap> levelMaps(levels);

    // --- Level 0: leaves on the padded domain ---
    glm::uvec3 padded = paddedGridSize;
    for (uint32_t z = 0; z < padded.z; ++z) {
        for (uint32_t y = 0; y < padded.y; ++y) {
            for (uint32_t x = 0; x < padded.x; ++x) {
                glm::uvec3 coord(x, y, z);
                uint8_t val = 0;
                if (x < originalGridSize.x && y < originalGridSize.y && z < originalGridSize.z) {
                    val = voxelValue(coord);
                }
                if (val != 0u) {
                    nodes.emplace_back(0, coord, val);
                    levelMaps[0][coord] = static_cast<uint32_t>(nodes.size() - 1);
                }
            }
        }
    }

    // --- Build upper levels from existing children only (sparse) ---
    for (int L = 1; L < levels; ++L) {
        // First pass: collect all parent coordinates and their children
        std::map<glm::uvec3, std::vector<uint32_t>, UVec3Comparator> parentChildrenMap;
        
        for (const auto& entry : levelMaps[L - 1]) {
            uint32_t childIdx = entry.second;
            // Store the child index for later processing
            const Node& child = nodes[childIdx];
            glm::uvec3 parentCoord = child.coord >> 1u;
            parentChildrenMap[parentCoord].push_back(childIdx);
        }

        // Second pass: create parent nodes and link children
        for (const auto& entry : parentChildrenMap) {
            const glm::uvec3& parentCoord = entry.first;
            const std::vector<uint32_t>& childrenIndices = entry.second;

            // Create parent node
            nodes.emplace_back(L, parentCoord, 0u);
            uint32_t parentIdx = static_cast<uint32_t>(nodes.size() - 1);
            levelMaps[L][parentCoord] = parentIdx;
            Node& parent = nodes[parentIdx];

            // Link children to parent
            for (uint32_t childIdx : childrenIndices) {
                const Node& child = nodes[childIdx];
                glm::uvec3 childOffset = child.coord & glm::uvec3(1u);
                uint32_t childSlot = childOffset.x | (childOffset.y << 1u) | (childOffset.z << 2u);

                parent.childrenMask |= (1u << childSlot);
                parent.children[childSlot] = static_cast<int32_t>(childIdx);
                nodes[childIdx].parentIndex = static_cast<int32_t>(parentIdx);
            }
        }

        // Compute parent colors
        for (const auto& entry : levelMaps[L]) {
            uint32_t pIdx = entry.second;
            Node& p = nodes[pIdx];

            std::array<int, 256> counts;
            counts.fill(0);
            for (int i = 0; i < 8; ++i) {
                int32_t cidx = p.children[i];
                if (cidx >= 0) {
                    counts[nodes[cidx].color]++;
                }
            }
            int best = 0;
            uint8_t bestColor = 0;
            for (int c = 0; c < 256; ++c) {
                if (counts[c] > best) {
                    best = counts[c];
                    bestColor = static_cast<uint8_t>(c);
                }
            }
            p.color = bestColor;
        }
    }
}


void SVO::flattenTree()
{
    flatNodesGPU.clear();
    flatNodesGPU.reserve(nodes.size()); // rough reserve

    std::function<void(int32_t)> flatten = [&](int32_t nIdx) {
        if(nIdx < 0 || nIdx >= (int32_t)nodes.size()) return;
        Node& n = nodes[nIdx];

        glm::vec3 mn, mx;
        computeWorldAABB(n, mn, mx);

        SVONodeGPU gn;
        gn.lowerCorner = mn;
        gn.upperCorner = mx;
        gn.colorIndex = n.color;
        gn.level = n.level;

        n.flatIndex = static_cast<int32_t>(flatNodesGPU.size());
        flatNodesGPU.push_back(gn);

        // traverse children in canonical order 0..7
        for(int i = 0; i < 8; ++i) {
            int32_t c = n.children[i];
            if(c >= 0) flatten(c);
        }
        };

    // find root candidates (nodes without parent)
    for(size_t i = 0; i < nodes.size(); ++i) {
        if(nodes[i].parentIndex == -1) {
            flatten(static_cast<int32_t>(i));
        }
    }
}


void SVO::computeWorldAABB(const Node& node, glm::vec3& outMin, glm::vec3& outMax) const
{
    // node.coord is an index in the padded grid at this node.level.
    // nodeSize = size of a node at level expressed in padded voxels * voxelSize
    float scale = static_cast<float>(1u << node.level); // how many padded voxels per node side
    glm::vec3 nodeSize = voxelSize * scale; // voxelSize computed with paddedGridSize
    outMin = worldLower + glm::vec3(node.coord) * nodeSize;
    outMax = outMin + nodeSize;

    // clamp to original world bounds to avoid extending into padded empty space
    outMin = glm::max(outMin, worldLower);
    outMax = glm::min(outMax, worldUpper);
}


std::vector<uint32_t> SVO::selectNodes(const glm::vec3& cameraPos, float lodBaseDist) const
{
    std::vector<uint32_t> result;
    result.reserve(1024);

    std::function<void(int32_t)> selectRec = [&](int32_t nIdx) {
        if(nIdx < 0 || nIdx >= (int32_t)nodes.size()) return;
        const Node& n = nodes[nIdx];

        glm::vec3 mn, mx;
        computeWorldAABB(n, mn, mx);
        glm::vec3 center = (mn + mx) * 0.5f;
        float dist = glm::length(cameraPos - center);
        glm::vec3 ext = mx - mn;
        float nodeExtent = std::max(std::max(ext.x, ext.y), ext.z);

        if(n.level == 0 || dist > lodBaseDist * nodeExtent) {
            if(n.flatIndex >= 0) result.push_back(static_cast<uint32_t>(n.flatIndex));
            return;
        }

        // descend
        for(int i = 0; i < 8; ++i) {
            int32_t c = n.children[i];
            if(c >= 0) selectRec(c);
        }
        };

    // start from roots
    for(size_t i = 0; i < nodes.size(); ++i) {
        if(nodes[i].parentIndex == -1) selectRec(static_cast<int32_t>(i));
    }

    return result;
}


size_t SVO::estimateMemoryUsageBytes() const
{
    size_t memory = 0;

    // Node memory
    memory += nodes.size() * sizeof(Node);

    // GPU nodes memory
    memory += flatNodesGPU.size() * sizeof(SVONodeGPU);

    return memory;
}