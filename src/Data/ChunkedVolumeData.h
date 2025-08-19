#pragma once

#include <Core/vk_types.h>
#include <vector>

#include "ChunkIntervalTree.h"

class VulkanEngine;

struct VolumeChunk
{
	glm::uvec3 chunkIndex; // chunk's xyz index in the grid in chunk elements starting from 0. (For example [2, 3, 1] means this chunk is 3rd in x, 4th in y and 2nd in z). This is used to compute chunk's position in the whole unit grid.
	glm::uvec3 startIndex; // Starting index of the chunk in the whole data. Used to offset into the correct position the whole buffer.
	size_t chunkFlatIndex; // index of the chunk in the chunks array
	size_t stagingBufferOffset; // offset (in bytes) in the staging buffer that holds all the chunks
	float minIsoValue, maxIsoValue; // among all the voxels in the chunk
	glm::vec3 lowerCornerPos, upperCornerPos; // Precomputed and stored. Could be computed on the fly as well
	std::vector<std::pair<float, size_t>> isoValueHistogram; // holds number of triangles that would be emitted per some predefined isovalues. Will be used to sort render chunks.
};

/*
	3D Volume Data Grid that is hold within chunks. All the chunks are extracted from the given data and put into a big staging buffer (in memory it looks like [chunk1, chunk2, ...] )

	The grid is a unit cube centered at origin (in range [-0.5, 0.5]) regardless of the resolution in any axis. 
*/
class ChunkedVolumeData
{
public:
	ChunkedVolumeData() = delete;
	ChunkedVolumeData(VulkanEngine* engine, const std::vector<float>& volumeData, const glm::uvec3& gridSize_in, const glm::uvec3& chunkSize_in, const glm::vec3& gridLowerCornerPos_in, const glm::vec3& gridUpperCornerPos_in);
	std::vector<VolumeChunk*> query(float isoValue) const;
	void destroyStagingBuffer(VulkanEngine* engine);
	glm::uvec3 getNumChunks() const;
	glm::uvec3 getChunkSize() const;
	size_t getNumChunksFlat() const;
	VkBuffer getStagingBuffer() const;
	void* getStagingBufferBaseAddress() const;
	size_t getTotalNumPointsPerChunk() const;
	glm::uvec3 getShellSize() const;
	const std::vector<VolumeChunk>& getChunks() const;
	void computeChunkIsoValueHistograms(float minIsoValue, float maxIsoValue, size_t numBins);
	size_t estimateNumTriangles(const VolumeChunk& chunk, float isoValue) const;
	~ChunkedVolumeData();
private:
	void extractChunkData(const std::vector<float>& volumeData, size_t chunkFlatIndex, VolumeChunk& chunk);
private:
	VulkanEngine* pEngine;
	std::vector<VolumeChunk> chunks;
	AllocatedBuffer chunksStagingBuffer;
	float* pChunksStagingBuffer; // mapped pointer of the staging buffer
	glm::uvec3 gridSize;
	glm::uvec3 chunkSize;
	glm::uvec3 numChunks;
	ChunkIntervalTree intervalTree;
	glm::vec3 gridLowerCornerPos;
	glm::vec3 gridUpperCornerPos;
};