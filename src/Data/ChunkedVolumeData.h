#pragma once

#include <Core/vk_types.h>
#include <vector>

#include "ChunkIntervalTree.h"

class VulkanEngine;

struct VolumeChunk
{
	glm::uvec3 chunkIndex; // chunk's xyz index in the grid in chunk elements starting from 0. (For example [2, 3, 1] means this chunk is 3rd in x, 4th in y and 2nd in z). This is used to compute chunk's position in the whole unit grid.
	size_t stagingBufferOffset; // offset (in bytes) in the staging buffer that holds all the chunks
	float minIsoValue, maxIsoValue; // among all the voxels in the chunk
};

/*
	3D Volume Data Grid that is hold within chunks. All the chunks are extracted from the given data and put into a big staging buffer (in memory it looks like [chunk1, chunk2, ...] )

	The grid is a unit cube centered at origin (in range [-0.5, 0.5]) regardless of the resolution in any axis. 
*/
class ChunkedVolumeData
{
public:
	ChunkedVolumeData(VulkanEngine* engine, const std::vector<float>& volumeData, glm::uvec3 gridSize_in, glm::uvec3 chunkSize_in);
	std::vector<VolumeChunk*> query(float isoValue) const;
	~ChunkedVolumeData();
private:
	void extractChunkData(const std::vector<float>& volumeData, size_t flatChunkIndex, VolumeChunk& chunk);
private:
	VulkanEngine* pEngine;
	std::vector<VolumeChunk> chunks;
	AllocatedBuffer chunksStagingBuffer;
	float* pChunksStagingBuffer; // mapped pointer of the staging buffer
	glm::uvec3 gridSize;
	glm::uvec3 chunkSize;
	glm::uvec3 numChunks;
	ChunkIntervalTree intervalTree;
};