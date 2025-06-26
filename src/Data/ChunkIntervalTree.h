#pragma once

struct VolumeChunk;

class ChunkIntervalTree
{
public:
    void build(const std::vector<VolumeChunk*>& chunks);

    std::vector<VolumeChunk*> query(float isoValue) const;
private:
    struct Node
    {
        float center;
        std::vector<VolumeChunk*> chunksLeft;    // Chunks completely left of center (max < center)
        std::vector<VolumeChunk*> chunksOverlap;  // Chunks that straddle center (min <= center <= max)
        std::vector<VolumeChunk*> chunksRight;   // Chunks completely right of center (min > center)
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        void build(const std::vector<VolumeChunk*>& chunks);

        void query(float isoValue, std::vector<VolumeChunk*>& result) const;
    };
private:
    std::unique_ptr<Node> root;
};
