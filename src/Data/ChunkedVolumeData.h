#pragma once

#include <Core/vk_types.h>
#include <vector>

#include "ChunkIntervalTree.h"

#include <algorithm>
#include <execution>

/*
	This .h file won't be including vk_engine as it would create a circular dependency where: engine->scene->chunkedVolumData->engine. So, I am only forward declaring.
	The scene that includes it will include vk_engine.h in its cpp file (translation unit) so the function definition will be visible during compilation
*/
class VulkanEngine;

struct VolumeChunk
{
	glm::uvec3 chunkIndex; // chunk's xyz index in the grid in chunk elements starting from 0. (For example [2, 3, 1] means this chunk is 3rd in x, 4th in y and 2nd in z). This is used to compute chunk's position in the whole unit grid.
	size_t chunkFlatIndex; // index of the chunk in the chunks array
	size_t stagingBufferOffset; // offset (in bytes) in the staging buffer that holds all the chunks
	float minIsoValue, maxIsoValue; // among all the voxels in the chunk
	glm::vec3 lowerCornerPos, upperCornerPos; // Precomputed and stored. Could be computed on the fly as well
};

/*
	3D Volume Data Grid that is hold within chunks. All the chunks are extracted from the given data and put into a big staging buffer (in memory it looks like [chunk1, chunk2, ...] )

	The grid is a unit cube centered at origin (in range [-0.5, 0.5]) regardless of the resolution in any axis. 
*/
template<typename T>
class ChunkedVolumeData
{
public:
	ChunkedVolumeData() = delete;
	ChunkedVolumeData(VulkanEngine* engine, const std::vector<T>& volumeData, const glm::uvec3& gridSize_in, const glm::uvec3& chunkSize_in, const glm::vec3& gridLowerCornerPos_in, const glm::vec3& gridUpperCornerPos_in, bool buildIntervalTree = true)
		:
		gridSize(gridSize_in),
		chunkSize(chunkSize_in),
		gridLowerCornerPos(gridLowerCornerPos_in),
		gridUpperCornerPos(gridUpperCornerPos_in)
	{
		pEngine = engine;
		/*
			Chunk size determines how many points on each axis of a chunk. Each point corresponds to the top-left-back point of the voxel. So, chunk size of n means, n top-left-back points and thus n voxels.
			However, for the bottom-right-front boundary, right neighbouring value is needed to be able reconstruct triangles in that voxel so that value is also read, thus +1.
			Moreover, to be able to compute normals consistently (as I am using forward differences for normals), the right neighbouring value of that right neighbour is needed, thus another +1.
			So, each chunk actualy contains a +2 shell on their bottom-right-front boundaries for correct reconstruction.

			Note: the last +1 for the normals could be alleviated by computing the normals using backward differences on the boundaries. For consistency, I am having +2.
		*/
		numChunks = (gridSize + chunkSize - 1u) / chunkSize;
		size_t numChunksFlat = numChunks.z * numChunks.y * numChunks.x;
		size_t numPointsPerChunk = (chunkSize.z + 2) * (chunkSize.y + 2) * (chunkSize.x + 2);

		// Allocate the staging buffer
		size_t stagingBufferSize = numChunksFlat * numPointsPerChunk * sizeof(T);
		chunksStagingBuffer = pEngine->createBuffer(stagingBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
		pChunksStagingBuffer = (T*)pEngine->getMappedStagingBufferData(chunksStagingBuffer);
		std::memset(pChunksStagingBuffer, 0, stagingBufferSize);

		chunks.resize(numChunksFlat);
		std::vector<size_t> indices(numChunksFlat);
		std::iota(indices.begin(), indices.end(), 0);
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t idx) {
			size_t z = idx / (numChunks.x * numChunks.y);
			size_t y = (idx / numChunks.x) % numChunks.y;
			size_t x = idx % numChunks.x;

			VolumeChunk chunk;
			chunk.chunkIndex = glm::uvec3(x, y, z);
			extractChunkData(volumeData, idx, chunk);

			chunks[idx] = std::move(chunk);
			});

		// Construct the interval tree
		std::vector<VolumeChunk*> chunkAddresses(numChunksFlat);
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
			chunkAddresses[i] = &chunks[i];
			});

		if(buildIntervalTree)
		{
			intervalTree.build(chunkAddresses);
			intervalTreeBuilt = true;
		}
	}
	std::vector<VolumeChunk*> query(float isoValue) const
	{
		if(intervalTreeBuilt)
		{
			return intervalTree.query(isoValue);
		}
		else
		{
			fmt::println("Before querying make sure that interval tree was build with buildIntervalTree flag set in the constructor");
			return {};
		}
	}
	void destroyStagingBuffer()
	{
		pEngine->destroyBuffer(chunksStagingBuffer);
		chunksStagingBuffer.buffer = VK_NULL_HANDLE;
	}
	glm::uvec3 getNumChunks() const
	{
		return numChunks;
	}
	glm::uvec3 getChunkSize() const
	{
		return chunkSize;
	}
	size_t getNumChunksFlat() const
	{
		return chunks.size();
	}
	VkBuffer getStagingBuffer() const
	{
		return chunksStagingBuffer.buffer;
	}
	void* getStagingBufferBaseAddress() const
	{
		return pChunksStagingBuffer;
	}
	size_t getTotalNumPointsPerChunk() const
	{
		return (chunkSize.x + 2) * (chunkSize.y + 2) * (chunkSize.z + 2);
	}
	glm::uvec3 getShellSize() const
	{
		return chunkSize + 2u;
	}
	const std::vector<VolumeChunk>& getChunks() const
	{
		return chunks;
	}
	~ChunkedVolumeData()
	{
		if(chunksStagingBuffer.buffer != VK_NULL_HANDLE)
		{
			pEngine->destroyBuffer(chunksStagingBuffer);
		}
	}
private:
	void extractChunkData(const std::vector<T>& volumeData, size_t chunkFlatIndex, VolumeChunk& chunk)
	{
		chunk.chunkFlatIndex = chunkFlatIndex;

		// Compute the bound indices of the chunk for fetching the data from volume data. (Mapping from chunk index to actual grid index)
		glm::uvec3 startIndex = chunkSize * chunk.chunkIndex;
		glm::uvec3 endIndex = glm::min(startIndex + chunkSize + 2u, gridSize); // exclusive end

		// Compute the lower and upper corner positions of the chunk from the grid corners.
		glm::vec3 stepSize = (gridUpperCornerPos - gridLowerCornerPos) / glm::vec3(gridSize - 1u);
		chunk.lowerCornerPos = gridLowerCornerPos + glm::vec3(startIndex) * stepSize;
		chunk.upperCornerPos = chunk.lowerCornerPos + glm::vec3(chunkSize) * stepSize;

		chunk.minIsoValue = FLT_MAX;
		chunk.maxIsoValue = -FLT_MAX;

		// Compute and store the offset of the given chunk in the staging buffer
		size_t chunkVoxelsFlatIndex = chunkFlatIndex * ((chunkSize.x + 2) * (chunkSize.y + 2) * (chunkSize.z + 2)); // the actual starting index of the voxels of the chunk in the staging buffer
		chunk.stagingBufferOffset = chunkVoxelsFlatIndex * sizeof(T);

		T* pChunkVoxels = pChunksStagingBuffer + chunkVoxelsFlatIndex;

		glm::uvec3 pointsPerChunk = chunkSize + 2u;

		for(size_t z = startIndex.z; z < endIndex.z; ++z)
		{
			for(size_t y = startIndex.y; y < endIndex.y; ++y)
			{
				for(size_t x = startIndex.x; x < endIndex.x; ++x)
				{
					T val = volumeData[x + gridSize.x * (y + gridSize.y * z)];
					if(intervalTreeBuilt)
					{
						// In the cases where we need interval tree, we always use compressed uint8_t data
						float decompressedVal = val / 255.0f;
						chunk.minIsoValue = std::min(chunk.minIsoValue, decompressedVal);
						chunk.maxIsoValue = std::max(chunk.maxIsoValue, decompressedVal);
					}
					// Write the value into the correct position in the staging buffer
					glm::uvec3 localIdx = glm::uvec3(x, y, z) - startIndex;
					pChunkVoxels[localIdx.x + pointsPerChunk.x * (localIdx.y + pointsPerChunk.y * localIdx.z)] = val;
				}
			}
		}
	}
private:
	VulkanEngine* pEngine;
	std::vector<VolumeChunk> chunks;
	AllocatedBuffer chunksStagingBuffer;
	T* pChunksStagingBuffer; // mapped pointer of the staging buffer
	glm::uvec3 gridSize;
	glm::uvec3 chunkSize;
	glm::uvec3 numChunks;
	ChunkIntervalTree intervalTree;
	glm::vec3 gridLowerCornerPos;
	glm::vec3 gridUpperCornerPos;
	bool intervalTreeBuilt;
};