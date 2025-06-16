#include "ChunkIntervalTree.h"

#include "ChunkedVolumeData.h"

#include <algorithm>

void ChunkIntervalTree::build(const std::vector<VolumeChunk*>& chunks)
{
    root = std::make_unique<Node>();
    root->build(chunks);
}

std::vector<VolumeChunk*> ChunkIntervalTree::query(float isoValue) const
{
    std::vector<VolumeChunk*> result;
    if(root)
    {
        root->query(isoValue, result);
    }
    return result;
}

void ChunkIntervalTree::Node::build(const std::vector<VolumeChunk*>& chunks)
{
    if(chunks.empty()) return;

    // Collect all endpoints to find median
    std::vector<float> endpoints;
    for(const auto& chunk : chunks)
    {
        endpoints.push_back(chunk->minIsoValue);
        endpoints.push_back(chunk->maxIsoValue);
    }
    std::sort(endpoints.begin(), endpoints.end());
    center = endpoints[endpoints.size() / 2];

    // Partition chunks into three categories
    for(const auto& chunk : chunks)
    {
        if(chunk->maxIsoValue < center)
        {
            chunksLeft.push_back(chunk);
        }
        else if(chunk->minIsoValue > center)
        {
            chunksRight.push_back(chunk);
        }
        else
        {
            chunksOverlap.push_back(chunk);
        }
    }

    // Recursively build subtrees if needed
    if(!chunksLeft.empty())
    {
        left = std::make_unique<Node>();
        left->build(chunksLeft);
    }
    if(!chunksRight.empty())
    {
        right = std::make_unique<Node>();
        right->build(chunksRight);
    }
}

void ChunkIntervalTree::Node::query(float isoValue, std::vector<VolumeChunk*>& result) const
{
    if(isoValue < center)
    {
        // Add all overlapping chunks where min <= isoValue
        for(const auto& chunk : chunksOverlap)
        {
            if(chunk->minIsoValue <= isoValue)
            {
                result.push_back(chunk);
            }
        }
        // Search left subtree
        if(left) left->query(isoValue, result);
    }
    else if(isoValue > center)
    {
        // Add all overlapping chunks where max >= isoValue
        for(const auto& chunk : chunksOverlap)
        {
            if(chunk->maxIsoValue >= isoValue)
            {
                result.push_back(chunk);
            }
        }
        // Search right subtree
        if(right) right->query(isoValue, result);
    }
    else
    {
        // isoValue == center, add all overlapping chunks
        result.insert(result.end(), chunksOverlap.begin(), chunksOverlap.end());
    }
}
