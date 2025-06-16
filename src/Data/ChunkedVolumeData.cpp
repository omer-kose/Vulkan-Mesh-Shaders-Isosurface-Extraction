#include "ChunkedVolumeData.h"

#include <Core/vk_engine.h>

ChunkedVolumeData::ChunkedVolumeData(VulkanEngine* engine, const std::vector<float>& volumeData, glm::uvec3 gridSize_in, glm::uvec3 chunkSize_in)
	:
	gridSize(gridSize_in),
	chunkSize(chunkSize_in)
{
	pEngine = engine;
	numChunks = (gridSize + chunkSize - glm::uvec3(1, 1, 1)) / chunkSize;
	size_t numChunksFlat = numChunks.z * numChunks.y * numChunks.x;
	size_t numVoxelsPerChunk = chunkSize.z * chunkSize.y * chunkSize.x;
	chunks.reserve(numChunksFlat);

	// Allocate the staging buffer
	size_t stagingBufferSize = numChunksFlat * numVoxelsPerChunk * sizeof(float);
	chunksStagingBuffer = pEngine->createBuffer(stagingBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	pChunksStagingBuffer = (float*)pEngine->getMappedStagingBufferData(chunksStagingBuffer);
	/*
		Set all to FLT_MAX. This guarantees that, the chunks that is partly outside of the grid (if the size along some axis is not multiple of chunk size) get a default maximum value 
		(they are infinitely far away from the isosurface and not taken into account while computing min and max isovalues).
	*/
	for(size_t i = 0; i < numChunksFlat * numVoxelsPerChunk; ++i)
	{
		pChunksStagingBuffer[i] = FLT_MAX;
	}

	// Divide volume into chunks
	for(size_t z = 0; z < numChunks.z; ++z)
	{
		for(size_t y = 0; y < numChunks.y; ++y)
		{
			for(size_t x = 0; x < numChunks.x; ++x)
			{
				VolumeChunk chunk;
				chunk.chunkIndex = glm::uvec3(x, y, z);
				extractChunkData(volumeData, x + numChunks.x * (y + numChunks.y * z), chunk);
				chunks.push_back(chunk);
			}
		}
	}

	// Construct the interval tree
	std::vector<VolumeChunk*> chunkAddresses(numChunksFlat);
	for(int i = 0; i < numChunksFlat; ++i)
	{
		chunkAddresses[i] = &chunks[i];
	}

	intervalTree.build(chunkAddresses);
}

std::vector<VolumeChunk*> ChunkedVolumeData::query(float isoValue) const
{
	return intervalTree.query(isoValue);
}

ChunkedVolumeData::~ChunkedVolumeData()
{
	pEngine->destroyBuffer(chunksStagingBuffer);
}

/*
	Extracts and writes the data of the chunk into the staging buffer
*/
void ChunkedVolumeData::extractChunkData(const std::vector<float>& volumeData, size_t flatChunkIndex, VolumeChunk& chunk)
{
	// Compute the bound indices of the chunk for fetching the data from volume data. (Mapping from chunk index to actual grid index)
	glm::uvec3 startIndex = chunkSize * chunk.chunkIndex;
	glm::uvec3 endIndex = glm::min(startIndex + chunkSize, gridSize);

	chunk.minIsoValue = FLT_MAX;
	chunk.maxIsoValue = -FLT_MAX;

	// Compute and store the offset of the given chunk in the staging buffer
	size_t chunkVoxelsFlatIndex = flatChunkIndex * (chunkSize.z * chunkSize.y * chunkSize.x); // the actual starting index of the voxels of the chunk in the staging buffer
	chunk.stagingBufferOffset = chunkVoxelsFlatIndex * sizeof(float);
	
	float* pChunkVoxels = pChunksStagingBuffer + chunkVoxelsFlatIndex;
	
	for(size_t z = startIndex.z; z < endIndex.z; ++z)
	{
		for(size_t y = startIndex.y; y < endIndex.y; ++y)
		{
			for(size_t x = startIndex.x; x < endIndex.x; ++x)
			{
				float val = volumeData[x + gridSize.x * (y + gridSize.y * z)];
				chunk.minIsoValue = std::min(chunk.minIsoValue, val);
				chunk.maxIsoValue = std::max(chunk.maxIsoValue, val);
				// Write the value into the correct position in the staging buffer
				pChunkVoxels[(x - startIndex.x) + chunkSize.x * ((y - startIndex.y) + chunkSize.y * (z - startIndex.z))] = val;
			}
		}
	}
}
